import ast
import json
import logging
import math
import os
import io
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
# import datasets as hfds
# from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
# from io import BytesIO
# import glob
# import h5py
# import xml.etree.ElementTree as ET

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

category_to_idx = {c: i for i, c in enumerate(object_categories)}

def read_split(root, dataset, split):
    base_path = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    filename = os.path.join(base_path, object_categories[0] + '_' + split + '.txt')

    with open(filename, 'r') as f:
        paths = []
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 0:
                assert len(line) == 2
                paths.append(line[0])

        return tuple(paths)


def read_bndbox(root, dataset, paths):
    xml_base = os.path.join(root, 'VOCdevkit', dataset, 'Annotations')
    instances = []
    for path in paths:
        xml = ET.parse(os.path.join(xml_base, path + '.xml'))
        for obj in xml.findall('object'):
            c = obj[0]
            assert c.tag == 'name', c.tag
            c = category_to_idx[c.text]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox[0].text)  # left
            ymin = int(bndbox[1].text)  # top
            xmax = int(bndbox[2].text)  # right
            ymax = int(bndbox[3].text)  # bottom
            instances.append((path, (xmin, ymin, xmax, ymax), c))
    return instances

class PASCALVoc2007Cropped(torch.utils.data.Dataset):
    """
    voc2007 is originally object detection and multi-label.
    In this version, we just convert it to single-label per image classification
    problem by looping over bounding boxes in the dataset and cropping the relevant
    object.
    """
    def __init__(self, root, set, transform=None, download=False, target_transform=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        
        # download dataset
        # if download:
        #     download_voc2007(self.root)

        paths = read_split(self.root, 'VOC2007', set)
        self.bndboxes = read_bndbox(self.root, 'VOC2007', paths)
        self.classes = object_categories

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of bndboxes=%d' % (
            set, len(self.classes), len(self.bndboxes)))

    def __getitem__(self, index):
        path, crop, target = self.bndboxes[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg'))
        img = img.crop(crop)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.bndboxes)

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights

def build_wds_dataset(args, dataset_name, transform, split="test", data_dir="root", cache_dir=None):
    """
    Load a dataset in WebDataset format. Either local paths or HTTP URLs can be specified.
    Expected file structure is:
    ```
    data_dir/
        train/
            nshards.txt
            0.tar
            1.tar
            ...
        test/
            nshards.txt
            0.tar
            1.tar
            ...
        classnames.txt
        zeroshot_classification_templates.txt
        dataset_type.txt
    ```
    Classnames and templates are required for zeroshot classification, while dataset type
    (equal to "retrieval") is required for zeroshot retrieval datasets.

    You can use the `clip_benchmark_export_wds` or corresponding API
    (`clip_benchmark.webdataset_builder.convert_dataset`) to convert datasets to this format.

    Set `cache_dir` to a path to cache the dataset, otherwise, no caching will occur.
    """
    import webdataset as wds

    def read_txt(fname):
        if "://" in fname:
            stream = os.popen("curl -L -s --fail '%s'" % fname, "r")
            value = stream.read()
            if stream.close():
                raise FileNotFoundError("Failed to retreive data")
        else:
            with open(fname, "r") as file:
                value = file.read()
        return value
    # Special handling for Huggingface datasets
    # Git LFS files have a different file path to access the raw data than other files
    if data_dir.startswith("https://huggingface.co/datasets"):
        # Format: https://huggingface.co/datasets/<USERNAME>/<REPO>/tree/<BRANCH>
        *split_url_head, _, url_path = data_dir.split("/", 7)
        url_head = "/".join(split_url_head)
        metadata_dir = "/".join([url_head, "raw", url_path])
        tardata_dir = "/".join([url_head, "resolve", url_path])
    else:
        metadata_dir = tardata_dir = data_dir
    # Get number of shards
    nshards_fname = os.path.join(metadata_dir, split, "nshards.txt")
    nshards = int(read_txt(nshards_fname)) # Do not catch FileNotFound, nshards.txt should be mandatory
    # Get dataset type (classification or retrieval)
    type_fname = os.path.join(metadata_dir, "dataset_type.txt")
    try:
        dataset_type = read_txt(type_fname).strip().lower()
    except FileNotFoundError:
        # print("WARNING: dataset_type.txt not found, assuming type=classification")
        dataset_type = "classification"
    #
    filepattern = os.path.join(tardata_dir, split, "{0..%d}.tar" % (nshards - 1))
    # Load webdataset (support WEBP, PNG, and JPG for now)
    if not cache_dir or not isinstance(cache_dir, str):
        cache_dir = None
    dataset = (
        wds.WebDataset(filepattern, cache_dir=cache_dir, nodesplitter=lambda src: src)
        .decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]))
    )
    # Load based on classification or retrieval task
    if dataset_type == "retrieval":
        dataset = (dataset
            .to_tuple(["webp", "png", "jpg", "jpeg"], "txt")
            .map_tuple(transform, str.splitlines)
        )
        dataset.classes = dataset.templates = None
    else:
        label_type = "npy" if dataset_type == "multilabel" else "cls" # Special case for multilabel
        dataset = (dataset
            .to_tuple(["webp","png","jpg","jpeg","0.webp"], label_type)
            .map_tuple(transform, None)
        )
        # Get class names if present
        classnames_fname = os.path.join(metadata_dir, "classnames.txt")
        try:
            dataset.classes = [line.strip() for line in read_txt(classnames_fname).splitlines()]
        except FileNotFoundError:
            print("WARNING: classnames.txt not found")
            dataset.classes = None
        # Get zeroshot classification templates if present
        templates_fname = os.path.join(metadata_dir, "zeroshot_classification_templates.txt")
        try:
            dataset.templates = [line.strip() for line in read_txt(templates_fname).splitlines()]
        except FileNotFoundError:
            print("WARNING: zeroshot_classification_templates.txt not found")
            dataset.templates = None

    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset.batched(args.batch_size),
        batch_size=None,
        num_workers=args.workers,
        sampler=sampler,
    )

    return dataloader

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        self.parquet_dir = parquet_dir
        parquet_files = glob.glob(os.path.join(parquet_dir, 'test*.parquet'))

        if not parquet_files:
            raise FileNotFoundError("no 'test'  .parquet ")

        dfs = [pd.read_parquet(parquet_file) for parquet_file in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_data = row['image']['bytes']
        image = Image.open(BytesIO(image_data))
        
        if 'birdsnap' in self.parquet_dir:
            label = torch.tensor(row['scientific'], dtype=torch.long)
        else:
            label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    # assert split in ["train", "val", "v2"]
    assert split in ["train", "val", "v2", "sketch", "a", "r", "o", "c"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    elif split == "sketch":
        data_path = args.imagenet_sketch
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)
    elif split == "a":
        data_path = args.imagenet_a
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)
    elif split == "o":
        data_path = args.imagenet_o
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)
    elif split == "c":
        data_path = args.imagenet_c
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)
    elif split == "r":
        data_path = args.imagenet_r
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    # print(preprocess_fn)
    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_flowers(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = datasets.Flowers102(root=args.flowers_102, transform=preprocess_fn, download=True, split=split)
    print(preprocess_fn)
    
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_food(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = datasets.Food101(root=args.food_101, transform=preprocess_fn, download=True, split=split)
    print(preprocess_fn)
    
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

# class MultiParquetDataset(Dataset):
#     def __init__(self, parquet_files, transform=None):
#         self.parquet_files = parquet_files
#         self.transform = transform
#         self.data = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_bytes = self.data.iloc[0]['image']['bytes']
#         label = self.data.iloc[idx]['label']
#         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

def get_stanford(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = datasets.StanfordCars(root=args.stanford, transform=preprocess_fn, download=False, split=split)
    print(preprocess_fn)
    # if split == "train":
    #     train_files = [os.path.join(args.stanford, f) for f in os.listdir(args.stanford) if f.startswith('train')]
    #     dataset = MultiParquetDataset(train_files, transform=preprocess_val)
    # else:
    #     test_files = [os.path.join(args.stanford, f) for f in os.listdir(args.stanford) if f.startswith('test')]
    #     dataset = MultiParquetDataset(test_files, transform=preprocess_val)
    
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_pets(args, preprocess_fns, split):
    assert split in ["train", "test"]
    preprocess_train, preprocess_val = preprocess_fns
    if split == "train":
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val
    if split == "train":
        split = "trainval"

    # dataset = datasets.OxfordIIITPet(root=args.pets, target_types='category', transform=preprocess_fn, download=True, split=split)
    # dataset = datasets.ImageFolder(args.pets, transform=preprocess_fn)
    if split == "trainval":
        train_path = os.path.join(args.pets, 'train')
        dataset = datasets.ImageFolder(train_path, transform=preprocess_fn)
    else:
        test_path = os.path.join(args.pets, 'test')
        dataset = datasets.ImageFolder(test_path, transform=preprocess_fn)
    print(preprocess_fn)
    
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_cifar(args, preprocess_fns, split, version):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    if version == '10':
        dataset = datasets.CIFAR10(root=args.cifar10, transform=preprocess_fn, download=True, train=is_train)
    elif version == '100':
        dataset = datasets.CIFAR100(root=args.cifar100, transform=preprocess_fn, download=True, train=is_train)
    print(preprocess_fn)

    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args["train_data"] if is_train else args["val_data"]
    assert input_shards is not None
    resampled = args.get('dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args["train_num_samples"] is not None:
            num_samples = args["train_num_samples"]
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args["val_num_samples"] or 0 
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args["train_data_upsampling_factors"] is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args["train_data_upsampling_factors"],
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args["seed"],
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args["batch_size"], partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args["workers"] * args["world_size"], 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args["batch_size"] * args["world_size"]
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args["workers"])
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args["batch_size"])

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args["workers"],
        persistent_workers=args["workers"] > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


class PromptTokenizeCaption:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, texts):
        texts = [f"a photo of {text}" for text in texts]
        return self.tokenizer(texts[:5])

class DocciPromptTokenizeCaption:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, texts):
        texts = [f"a photo of {text}" for text in texts]
        return self.tokenizer(texts)


def get_mscoco(args, preprocess_fn, tokenizer):

     dataset = datasets.CocoCaptions(
             root=args.ms_coco,
             annFile=args.ms_coco_annot,
             transform=preprocess_fn,
             target_transform=PromptTokenizeCaption(tokenizer)
             )

     return dataset


class Flickr(datasets.VisionDataset):
    def __init__(self, root, annFile, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform

        with open(annFile) as f:
            data = json.load(f)

        self.data = data
        self.root = root

    def __getitem__(self, index: int):
        image_name = self.data[index]['image']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        image = self.transform(image)

        captions = self.data[index]['caption']
        captions = self.target_transform(captions)

        return image, captions

    def __len__(self):
        return len(self.data)


def get_flickr(args, preprocess_fn, tokenizer):

     dataset = Flickr(
             root=args.flickr,
             annFile=args.flickr_annot,
             transform=preprocess_fn,
             target_transform=PromptTokenizeCaption(tokenizer)
             )

     return dataset

class DOCCIDataset(datasets.VisionDataset):
    def __init__(self, root, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform

        data = hfds.Dataset.from_file(root)
        
        # OLD_PREFIX = "/lpai/volumes/jfs-data-lhp-bd-ga/" 
        # NEW_PREFIX = "/lpai/volumes/so-data-lhp-bd-ga/"

        # def fix_path(example):
        #     if example["image"].startswith(OLD_PREFIX):
        #         example["image"] = example["image"].replace(OLD_PREFIX, NEW_PREFIX)
        #     return example

        # data = data.cast_column("image", Value("string"))

        # data = data.map(fix_path)

        # data = data.cast_column("image", Image())

        self.data = data
        self.root = root


    def __getitem__(self, index):
        image = self.data[index]["image"]
        image = self.transform(image)

        caption = self.data[index]["description"]
        captions = [s.strip() for s in caption.split('.') if s.strip()]
        captions = self.target_transform(captions)

        return image, captions

    def __len__(self):
        return len(self.data)

def get_docci(args, preprocess_fn, tokenizer):
    dataset = DOCCIDataset(
        root=args.docci,
        transform=preprocess_fn,
        target_transform=DocciPromptTokenizeCaption(tokenizer)
    )

    return dataset

def get_mnist(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    # dataset = ParquetImageDataset(parquet_dir=args.mnist, transform=preprocess_fn)
    # dataset = datasets.ImageFolder(args.mnist, transform=preprocess_fn)
    train_dataset = datasets.MNIST(root=args.mnist, train=True, transform=preprocess_fn, download=False)
    dataset = datasets.MNIST(root=args.mnist, train=False, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_caltech101(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = datasets.ImageFolder(args.caltech, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_fer2013(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    # dataset = ParquetImageDataset(parquet_dir=args.mnist, transform=preprocess_fn)
    dataset = datasets.ImageFolder(args.fer2013, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_imagefolder(path, args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = datasets.ImageFolder(path, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_zerocls(path, args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = ParquetImageDataset(parquet_dir=path, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)
def get_voc(args, preprocess_fns, split):
    assert split in ["train", "test"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns
    if is_train:
        preprocess_fn = preprocess_train
    else:
        preprocess_fn = preprocess_val

    dataset = PASCALVoc2007Cropped(root=args.voc, set=split, transform=preprocess_fn)
    print(preprocess_fn)
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
        data["imagenet-train"] = get_imagenet(args, preprocess_fns, "train")
    
    if args.cifar10 is not None:
        data["cifar10"] = get_cifar(args, preprocess_fns, "test", version='10')
        data["cifar10-train"] = get_cifar(args, preprocess_fns, "train", version='10')

    if args.cifar100 is not None:
        data["cifar100"] = get_cifar(args, preprocess_fns, "test", version='100')
        data["cifar100-train"] = get_cifar(args, preprocess_fns, "train", version='100')

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = build_wds_dataset(args, "imagenet_v2", transform=preprocess_val, split="test",data_dir=args.imagenet_v2)
    
    if args.imagenet_sketch is not None:
        data["imagenet-sketch"] = get_imagenet(args, preprocess_fns, "sketch")
    
    if args.imagenet_a is not None:
        data["imagenet-a"] = get_imagenet(args, preprocess_fns, "a")
    
    if args.imagenet_o is not None:
        data["imagenet-o"] = get_imagenet(args, preprocess_fns, "o")

    if args.imagenet_r is not None:
        data["imagenet-r"] = get_imagenet(args, preprocess_fns, "r")

    if args.flowers_102 is not None:
        data["flowers-102"] = get_flowers(args, preprocess_fns, "test")
        data["flowers-102-train"] = get_flowers(args, preprocess_fns, "train")
    
    if args.food_101 is not None:
        data["food-101"] = get_food(args, preprocess_fns, "test")
        data["food-101-train"] = get_food(args, preprocess_fns, "train")

    if args.stanford is not None:
        data["stanford"] = get_stanford(args, preprocess_fns, "test")
        data["stanford-train"] = get_stanford(args, preprocess_fns, "train")
    
    if args.pets is not None:
        data["pets"] = get_pets(args, preprocess_fns, "test")
        data["pets-train"] = get_pets(args, preprocess_fns, "train")

    if args.ms_coco is not None:
        data["ms-coco"] = get_mscoco(args, preprocess_val, tokenizer)
    
    if args.flickr is not None:
        data["flickr"] = get_flickr(args, preprocess_val, tokenizer)
    
    if args.docci is not None:
        data["docci"] = get_docci(args, preprocess_val, tokenizer)
    
    if args.mnist is not None:
        data["mnist"] = get_mnist(args, preprocess_fns, "test")
        # data["mnist-train"] = get_mnist(args, preprocess_fns, "train")
    
    if args.caltech is not None:
        # data["caltech"] = get_zerocls(args.caltech, args, preprocess_fns, "test")
        data["caltech"] = get_caltech101(args, preprocess_fns, "test")

    if args.sun397 is not None:
        data["sun397"] = get_zerocls(args.sun397, args, preprocess_fns, "test")

    if args.fgvc_aircraft is not None:
        # data["fgvc_aircraft"] = get_zerocls(args.fgvc_aircraft, args, preprocess_fns, "test")
        data["fgvc_aircraft"] = get_imagefolder(args.fgvc_aircraft, args, preprocess_fns, "test")
    
    if args.country211 is not None:
        # data["country211"] = get_zerocls(args.country211, args, preprocess_fns, "test")
        data["country211"] = get_imagefolder(args.country211, args, preprocess_fns, "test")
    
    if args.birdsnap is not None:
        data["birdsnap"] = get_zerocls(args.birdsnap, args, preprocess_fns, "test")
    
    if args.dtd is not None:
        data["dtd"] = get_zerocls(args.dtd, args, preprocess_fns, "test")

    if args.eurosat is not None:
        # data["eurosat"] = get_zerocls(args.eurosat, args, preprocess_fns, "test")
        data["eurosat"] = get_imagefolder(args.eurosat, args, preprocess_fns, "test")

    if args.fer2013 is not None:
        data["fer2013"] = get_fer2013(args, preprocess_fns, "test")

    if args.gtsrb is not None:
        # data["gtsrb"] = get_zerocls(args.gtsrb, args, preprocess_fns, "test")
        data["gtsrb"] = get_imagefolder(args.gtsrb, args, preprocess_fns, "test")
    
    if args.pcam is not None:
        # data["pcam"] = build_wds_dataset(args, "pcam", transform=preprocess_val, split="test",data_dir=args.pcam)
        data["pcam"] = get_imagefolder(args.pcam, args, preprocess_fns, "test")

    if args.rendered_sst2 is not None:
        # data["rendered_sst2"] = get_zerocls(args.rendered_sst2, args, preprocess_fns, "test")
        data["rendered_sst2"] = get_imagefolder(args.rendered_sst2, args, preprocess_fns, "test")
    
    if args.objectnet is not None:
        data["objectnet"] = build_wds_dataset(args, "objectnet", transform=preprocess_val, split="test",data_dir=args.objectnet)

    if args.resisc45 is not None:
        # data["resisc45"] = get_zerocls(args.resisc45, args, preprocess_fns, "test")
        data["resisc45"] = get_imagefolder(args.resisc45, args, preprocess_fns, "test")
    
    if args.stl10 is not None:
        data["stl10"] = get_zerocls(args.stl10, args, preprocess_fns, "test")

    if args.voc is not None:
        data["voc"] = get_voc(args, preprocess_fns, "test")

    return data
