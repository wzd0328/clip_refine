import argparse
import functools
import math
import os
import pdb
import random
import sys
import traceback

import ignite
import ignite.utils as ignite_utils
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from ignite.metrics.running_average import RunningAverage
from torch.backends import cudnn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import extensions

from evaluator.classification import ClassifierEvaluator
from evaluator.contrastive import ContrativeEvaluator
from util import train_util, yaml_utils, distributed


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("--------------")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("--------------")

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main(config):
    # random.seed(config["seed"])
    # np.random.seed(config["seed"])
    # torch.manual_seed(config["seed"])
    # device = "cuda" if (torch.cuda.is_available()) else "cpu"
    device = distributed.init_distributed_device(config)

    random_seed(config["seed"], 0)

    logger = ignite_utils.setup_logger(name=config["pattern"])
    log_basic_info(logger, config)

    # Create output directory
    logger.info("Creating output directory")
    out = os.path.join(config["results_dir"], config["pattern"], f"experiment{config['experiment_id']}")
    train_util.create_result_dir(out, config["config_path"])

    # Distributed info
    if config["distributed"]:
        logger.info(
            f'Running in distributed mode with multiple processes. Device: {config["device"]}.'
            f'Process (global: {config["rank"]}, local {config["local_rank"]}), total {config["world_size"]}.')
    else:
        logger.info(f'Running with a single process. Device {config["device"]}.')

    # Model
    logger.info("Constructing models")
    model = train_util.load_models(config["models"]["model"])
    # model = torch.nn.DataParallel(model)
    # model.to(device)

    if config["distributed"]:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config["local_rank"]],
            output_device=config["local_rank"],
            find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)

    random_seed(config["seed"], config["rank"])

    # DataLoader
    logger.info("Constructing data loaders")
    config["dataset"]["args"]["seed"] = config["seed"]
    config["dataset"]["args"]["workers"] = config["num_worker"]
    config["dataset"]["args"]["batch_size"] = config["batchsize"]
    config["dataset"]["args"]["epoch"] = config["epoch"]
    config["dataset"]["args"]["world_size"] = config.get("world_size", 1)
    config["dataset"]["args"]["transform"] = model.module.img_preprocess
    config["dataset"]["args"]["tokenizer"] = model.module.tokenizer
    train_loader = train_util.setup_pretraining_dataloaders(config)
    config["dataset_cls"]["args"]["transform"] = model.module.img_preprocess
    test_loader = train_util.setup_test_dataloader(config)
    max_iter = int(train_loader.num_batches * config["epoch"])

    # Optimizer
    logger.info("Constructing optimizers")
    opt = train_util.make_optimizer(model, config["optimizer"])

    if config["resume"]:
        logger.info("Resume training with snapshot:{}".format(config["resume"]))
        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(config["resume"])
            model.module.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            resume_step = checkpoint["optimizer"]["state"][0]["step"]

    # Updater
    logger.info("Constructing updater and evaluators")
    kwargs = config["updater"]["args"] if "args" in config["updater"] else {}
    kwargs.update(
        {
            "model": model,
            "optimizer": opt,
            "device": device,
            "train_loader": train_loader,
            "max_iteration": max_iter,
        }
    )
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    #print(f"updater.share_random_feat: {updater.share_random_feat}")
    if distributed.is_master(config):
        print(f"updater.teacher: {updater.teacher}")


    # Trainer := Ignite.Engine
    trainer = Engine(updater)
    monitoring_metrics = ["train_loss", "train_feat_gap"]

    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "train_loss")
    RunningAverage(output_transform=lambda x: x["feat_gap"]).attach(trainer, "train_feat_gap")
    if "log_metrics" in config:
        for key in config["log_metrics"]:
            monitoring_metrics.append(key)
            RunningAverage(output_transform=lambda x, key=key: x[key]).attach(trainer, key)

    logger.info(f"Monitoring Metrics: {monitoring_metrics}")
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # Contrastive Evaluator
    # cont_evaluator = Engine(ContrativeEvaluator(model=model, device=device))
    # RunningAverage(output_transform=lambda x: x["loss"]).attach(cont_evaluator, "val_loss")
    # RunningAverage(output_transform=lambda x: x["feat_gap"]).attach(cont_evaluator, "val_gap")
    # RunningAverage(output_transform=lambda x: x["alignment"]).attach(cont_evaluator, "val_alignment")
    # RunningAverage(output_transform=lambda x: x["uniformity"]).attach(cont_evaluator, "val_uniformity")

    # Classification Evaluator
    cls_evaluator = Engine(
        ClassifierEvaluator(
            model=model,
            device=device,
            classname_file=config["dataset_cls"]["classname_file"],
            tokenizer=model.module.tokenizer,
        )
    )
    RunningAverage(output_transform=lambda x: x["accuracy"]).attach(cls_evaluator, "accuracy")
    RunningAverage(output_transform=lambda x: x["loss"]).attach(cls_evaluator, "loss")

    # Event Handlers
    logger.info("Constructing event handlers")
    # Log Handler
    log = {"running": [{"epoch": "init"}], "best_val_loss": 10000000000000.0}
    log = log if not config["resume"] else extensions.load_log(out)
    logger_train = functools.partial(extensions.log_training_results_mps, log=log, pbar=pbar)
    # logger_val = functools.partial(
    #     extensions.log_pretraining_validation_results,
    #     cont_evaluator=cont_evaluator,
    #     val_loader=val_loader,
    #     cls_evaluator=cls_evaluator,
    #     test_loader=test_loader,
    #     log=log,
    #     pbar=pbar,
    #     dist=str(out),
    # )
    logger_val = functools.partial(
        extensions.log_pretraining_results,
        cls_evaluator=cls_evaluator,
        test_loader=test_loader,
        log=log,
        pbar=pbar,
        dist=str(out),
    )

    # Check Point Handler
    check_pointer = ModelCheckpoint(str(out), filename_prefix="model", n_saved=1, require_empty=not config["resume"])
    best_check_pointer = ModelCheckpoint(
        str(out),
        filename_prefix="best",
        score_function=extensions.check_cls_loss,
        n_saved=1,
        score_name="cls_loss",
        require_empty=not config["resume"],
    )

    # Learning Rate Schedule Handler
    lr_scheduler = train_util.set_learning_rate_scheduler(trainer, opt, config["optimizer"], max_iter)

    if config["resume"]:
        lr_scheduler.event_index = checkpoint["lr_scheduler"]["event_index"]

    # Append handlers to trainer/evaluator engine
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_val)
    # trainer.add_event_handler(
    #     Events.EPOCH_COMPLETED,
    #     check_pointer,
    #     {
    #         "model": model,
    #         "optimizer": opt,
    #         "lr_scheduler": lr_scheduler,
    #     },
    # )
    # cont_evaluator.add_event_handler(
    #     Events.COMPLETED,
    #     best_check_pointer,
    #     {
    #         "model": model,
    #     },
    # )
    cls_evaluator.add_event_handler(
        Events.COMPLETED,
        best_check_pointer,
        {
            "model": model,
        },
    )

    if config["resume"]:
        config["start_epoch"] = math.ceil(resume_step * config["batchsize"] / len(train_loader.dataset))
        resumer = functools.partial(extensions.resume_training, resume_epoch=config["start_epoch"])
        trainer.add_event_handler(Events.STARTED, resumer)

    if config["evaluate"]:
        trainer.add_event_handler(Events.STARTED, logger_val)

    # Run the training
    logger.info("Running train script")
    try:
        trainer.run(train_loader, max_epochs=config["epoch"])
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        log.update({"best_model": str(best_check_pointer.last_checkpoint)})
        extensions.dump_log(log, str(out))

    # Evaluation


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/base.yml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="/lpai/output/models/clip-refine/cc3m-openaiclip-vitL14-fix", help="directory to save the results to")
    parser.add_argument("--resume", type=str, default="", help="path to the snapshot")
    parser.add_argument("--experiment_id", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    config.config_dict["pattern"] = yaml_utils.make_pattern(config)
    config.config_dict.update(vars(args))
    main(config.config_dict)


if __name__ == "__main__":
    run()

