import pdb
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
from ignite.utils import convert_tensor

#from loss.mmd import MMDLoss


def distill(y_s, y_t, T=1.0, alpha=0.0):
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    gt = torch.eye(y_t.shape[0], device=y_t.device, dtype=torch.long)
    p_t = alpha * gt + (1.0 - alpha) * p_t
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T**2)
    return loss


def align(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class CLIPUpdater:
    def __init__(
        self,
        *args,
        lambda_cont=1.0,
        lambda_kd=None,
        distill_loss="fd",
        max_iteration=None,
        temperature=1.0,
        alpha_blending=0.0,
        use_amp=False,  # 添加AMP支持
        **kwargs,
    ):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.device = kwargs.pop("device")
        self.lambda_cont = lambda_cont
        self.max_iteration = max_iteration
        if lambda_kd is not None:
            self.lambda_kd = lambda_kd
            # Only deepcopy the underlying module, not the DDP wrapper
            # This avoids duplicating DDP state and reduces memory usage
            if hasattr(self.model, 'module'):
                # Model is wrapped in DDP or DataParallel
                self.teacher = deepcopy(self.model.module)
            else:
                self.teacher = deepcopy(self.model)
            self.teacher.fc_v = torch.nn.Identity()
            self.teacher.fc_t = torch.nn.Identity()
            self.teacher.to(self.device)
            self.loss_kd = self.select_kd_loss(distill_loss)
        else:
            self.teacher = None
        self.T = temperature
        self.alpha_blending = alpha_blending

        # 混合精度训练
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def select_kd_loss(self, distill_loss):
        if distill_loss == "fd":
            return self.fd_loss
        elif distill_loss == "ld":
            return self.ld_loss
        elif distill_loss == "kd":
            return self.kd_loss

    def get_batch(self, batch, device=None, non_blocking=True):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    def clip_loss(self, feat_i, feat_t, logit_scale):
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logit_scale * logits_per_image, labels)
            + F.cross_entropy(logit_scale * logits_per_text, labels)
        ) / 2
        return total_loss

    def kd_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_per_image_t = feat_it @ feat_tt.T
            logits_per_text_t = feat_tt @ feat_it.T
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        loss_kd = (
            distill(logits_per_image, logits_per_image_t.detach(), self.T, self.alpha_blending)
            + distill(logits_per_text, logits_per_text_t.detach(), self.T, self.alpha_blending)
        ) / 2
        return loss_kd

    def ld_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_per_image_t = feat_it @ feat_tt.T
            logits_per_text_t = feat_tt @ feat_it.T
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        loss_kd = (
            F.mse_loss(logits_per_image, logits_per_image_t.detach())
            + F.mse_loss(logits_per_text, logits_per_text_t.detach())
        ) / 2
        return loss_kd

    def fd_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
        loss_kd = F.mse_loss(feat_i, feat_it) + F.mse_loss(feat_t, feat_tt)
        return loss_kd

    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        images, texts = self.get_batch(batch, device=self.device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(images, texts.squeeze())
            feat_i, feat_t = out["image_features"], out["text_features"]
            contrastive_loss = self.clip_loss(feat_i, feat_t, out["logit_scale"])
            total_loss = self.lambda_cont * contrastive_loss

            if self.teacher:
                self.teacher.eval()  # Ensure teacher is in eval mode
                loss_kd = self.loss_kd(images, texts, feat_i, feat_t)
                total_loss = total_loss + self.lambda_kd * loss_kd
                report.update(
                    {
                        "loss_kd": loss_kd.detach().item(),
                    }
                )

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        feat_gap = F.pairwise_distance(feat_i, feat_t).mean()
        modality_gap = F.mse_loss(feat_i.mean(dim=-1), feat_t.mean(dim=-1))
        report.update(
            {
                "loss": contrastive_loss.detach().item(),
                "feat_gap": feat_gap.detach().item(),
                "modality_gap": modality_gap.detach().item(),
            }
        )
        return report

class ConcatUpdater(CLIPUpdater):
    def __init__(
        self,
        *args,
        # lambda_concat=1.0,
        lambda_struct=1.0,
        lambda_mid=1.0,
        regularization_decay=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.lambda_concat = lambda_concat
        self.lambda_struct = lambda_struct
        self.lambda_mid = lambda_mid
        self.regularization_decay = regularization_decay
        self.decay_rate = 1.0
    
    def update_decay_rate(self, current_iteration):
        if self.regularization_decay:
            assert current_iteration <= self.max_iteration
            self.decay_rate = 1.0 - (current_iteration / self.max_iteration)

    ## method2:按单模态维度concat，batch内对比损失，scale单独初始化---27.72
    # def concat_loss(self, images, texts, feat_i, logit_scale):
    #     with torch.no_grad():
    #         out_t = self.teacher(images, texts.squeeze())
    #         feat_it = out_t["image_features"]
    #     feat_i_concat = torch.cat([feat_i, feat_it], dim=1)
    #     feat_i_concat = F.normalize(feat_i_concat, dim=-1)
    #     logits_concat = feat_i_concat @ feat_i_concat.T
    #     labels = torch.arange(logits_concat.shape[0], device=self.device, dtype=torch.long)
    #     logits_concat = logits_concat - F.one_hot(labels, logits_concat.shape[1]) * logits_concat
    #     loss_kd = F.cross_entropy(logit_scale * logits_concat, labels)
    #     return loss_kd

    def concat_loss(self, images, texts, feat_i, feat_t, feat_s_concat):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]

        # feat_concat = torch.cat([feat_i, feat_t], dim=1) * feat_s_concat
        # feat_t_concat = torch.cat([feat_it, feat_tt], dim=1)
        # feat_concat = torch.cat([feat_i, feat_t], dim=1)
        # feat_concat = F.normalize(feat_concat, dim=-1)
        # feat_t_concat = F.normalize(feat_t_concat, dim=-1)

        # loss_concat = F.mse_loss(feat_s_concat, feat_it) + F.mse_loss(feat_s_concat, feat_tt)

        ## method1---60.632 , (去掉decay_rate)---60.61
        loss_kd = F.mse_loss(feat_i, feat_s_concat) + F.mse_loss(feat_t, feat_s_concat)

        return loss_kd

    ## method3:局部结构相似 ---60.242
    # def struct_loss(self, feat_i, feat_t, temperature=0.07):
    #     feat_i = F.normalize(feat_i, dim=-1)
    #     feat_t = F.normalize(feat_t, dim=-1)

    #     logits_img = feat_i @ feat_i.T
    #     logits_txt = feat_t @ feat_t.T

    #     kl_img_txt = distill(logits_img, logits_txt, temperature)
    #     kl_txt_img = distill(logits_txt, logits_img, temperature)

    #     loss_kd = (kl_img_txt + kl_txt_img) / 2

    #     return loss_kd

    ## method4:全局结构相似 ---59.04
    # def struct_loss(self, images, texts, feat_i, feat_t, temperature=0.07):
    #     with torch.no_grad():
    #         out_t = self.teacher(images, texts.squeeze())
    #         feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
    #         logits_imgt = feat_it @ feat_it.T
    #         logits_txtt = feat_tt @ feat_tt.T

    #     # feat_i = F.normalize(feat_i, dim=-1)
    #     # feat_t = F.normalize(feat_t, dim=-1)

    #     logits_img = feat_i @ feat_i.T
    #     logits_txt = feat_t @ feat_t.T

    #     kl_img_txt = distill(logits_img, logits_imgt, temperature)
    #     kl_txt_img = distill(logits_txt, logits_txtt, temperature)

    #     loss_kd = (kl_img_txt + kl_txt_img) / 2

    #     return loss_kd

    ## method5:全局结构相似+中间层特征 ---
    def struct_loss(self, images, texts, feat_i, feat_t, feat_mid_v=None, temperature=0.07):
        with torch.no_grad():
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_imgt = feat_it @ feat_it.T
            logits_txtt = feat_tt @ feat_tt.T

            feat_mid_vt = out_t["feat_mid_v"]
            feat_mid_vt = F.normalize(feat_mid_vt, dim=-1) if feat_mid_vt is not None else None

        # feat_i = F.normalize(feat_i, dim=-1)
        # feat_t = F.normalize(feat_t, dim=-1)

        logits_img = feat_i @ feat_i.T
        logits_txt = feat_t @ feat_t.T

        kl_img_txt = distill(logits_img, logits_imgt, temperature)
        kl_txt_img = distill(logits_txt, logits_txtt, temperature)

        loss_kd = (kl_img_txt + kl_txt_img) / 2

        loss_mid = F.mse_loss(feat_mid_v, feat_mid_vt) if feat_mid_vt is not None else 0.0

        return loss_kd, loss_mid

    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        self.update_decay_rate(engine.state.iteration)
        images, texts = self.get_batch(batch, device=self.device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(images, texts.squeeze())
            feat_i, feat_t = out["image_features"], out["text_features"]
            contrastive_loss = self.clip_loss(feat_i, feat_t, out["logit_scale"])
            loss_main = self.lambda_cont * contrastive_loss

            ## method1
            # loss_concat = self.concat_loss(images, texts, feat_i, feat_t, out["feat_st_concat"])
            # loss_concat = self.concat_loss(images, texts, feat_i, out["concat_logit_scale"])
            # total_loss = loss_main + self.decay_rate * self.lambda_concat * loss_concat
            # total_loss = loss_main + self.lambda_concat * loss_concat

            ## method3
            # loss_struct = self.struct_loss(images, texts, feat_i, feat_t)
            # total_loss = loss_main + self.decay_rate * self.lambda_struct * loss_struct

            ## method5
            loss_struct, loss_mid = self.struct_loss(images, texts, feat_i, feat_t, out["feat_mid_v"])
            total_loss = loss_main + self.lambda_struct * loss_struct + self.lambda_mid * loss_mid

            if self.teacher:
                self.teacher.eval()  # Ensure teacher is in eval mode
                loss_kd = self.loss_kd(images, texts, feat_i, feat_t)
                total_loss = total_loss + self.lambda_kd * loss_kd
                report.update(
                    {
                        "loss_kd": loss_kd.detach().item(),
                    }
                )

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        feat_gap = F.pairwise_distance(feat_i, feat_t).mean()
        modality_gap = F.mse_loss(feat_i.mean(dim=-1), feat_t.mean(dim=-1))
        report.update(
            {
                "loss": contrastive_loss.detach().item(),
                # "loss_concat": loss_concat.detach().item(),
                "loss_struct": loss_struct.detach().item(),
                "loss_mid": loss_mid.detach().item(),
                "feat_gap": feat_gap.detach().item(),
                "modality_gap": modality_gap.detach().item(),
            }
        )
        return report

class RandRegUpdater(CLIPUpdater):

    def __init__(
        self,
        *args,
        lambda_rand=1.0,
        strategy="std_sample",
        share_random_feat=True,
        mu=0.0,
        sigma=1.0,
        precomputed_stats=None,
        regularization_decay=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_rand = lambda_rand
        self.strategy = strategy
        self.share_random_feat = share_random_feat
        if self.strategy == "std_sample":
            self.mean, self.std = torch.tensor([mu]).to(self.device), torch.tensor([sigma]).to(self.device)
            self.random_dist = dists.Normal(loc=torch.tensor([mu]), scale=torch.tensor([sigma]))
        elif self.strategy in ["uniform_sample", "uniform_fixed"]:
            self.mean, self.std = torch.tensor([mu]).to(self.device), torch.tensor([sigma]).to(self.device)
            self.random_dist = dists.Uniform(low=torch.tensor([mu]), high=torch.tensor([sigma]))
        elif self.strategy in ["precomputed_fixed", "precomputed_sample"]:
            assert precomputed_stats is not None
            stats = np.load(precomputed_stats)
            self.mean, self.std = torch.from_numpy(stats["mean"]).to(self.device), torch.from_numpy(stats["std"]).to(
                self.device
            )
            self.random_dist = dists.Normal(loc=self.mean, scale=self.std)
        self.feature_loss_fn = F.mse_loss
        self.regularization_decay = regularization_decay
        self.decay_rate = 1.0

    def generate_random_feature(self, size):
        if self.strategy == "std_sample":
            f_rand = self.random_dist.sample(size).to(self.device)
        elif self.strategy == "uniform_sample":
            f_rand = self.random_dist.sample(size).to(self.device)
        elif self.strategy == "precomputed_sample":
            f_rand = self.random_dist.sample([size[0]]).to(self.device)
        elif self.strategy in ["precomputed_fixed", "uniform_fixed"]:
            f_rand = self.mean
        else:
            raise NotImplementedError
        return f_rand.squeeze()

    def update_decay_rate(self, current_iteration):
        if self.regularization_decay:
            assert current_iteration <= self.max_iteration
            self.decay_rate = 1.0 - (current_iteration / self.max_iteration)

    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        self.update_decay_rate(engine.state.iteration)
        images, texts = self.get_batch(batch, device=self.device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(images, texts.squeeze())
            feat_i, feat_t = out["image_features"], out["text_features"]
            contrastive_loss = self.clip_loss(feat_i, feat_t, out["logit_scale"])
            loss_main = self.lambda_cont * contrastive_loss
            if self.share_random_feat:
                feat_ip = feat_tp = self.generate_random_feature(feat_i.size())
            else:
                feat_ip = self.generate_random_feature(feat_i.size())
                feat_tp = self.generate_random_feature(feat_tp.size())
            loss_feat = self.feature_loss_fn(feat_i, feat_ip) + self.feature_loss_fn(feat_t, feat_tp)
            total_loss = loss_main + self.decay_rate * self.lambda_rand * loss_feat

            if self.teacher:
                self.teacher.eval()  # Ensure teacher is in eval mode
                loss_kd = self.loss_kd(images, texts, feat_i, feat_t)
                total_loss = total_loss + self.lambda_kd * loss_kd
                report.update(
                    {
                        "loss_kd": loss_kd.detach().item(),
                    }
                )

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        feat_gap = F.pairwise_distance(feat_i, feat_t).mean()
        modality_gap = F.mse_loss(feat_i.mean(dim=0), feat_t.mean(dim=0))
        report.update(
            {
                "loss": contrastive_loss.detach().item(),
                "loss_feat": loss_feat.detach().item(),
                "feat_gap": feat_gap.detach().item(),
                "modality_gap": modality_gap.detach().item(),
            }
        )
        return report

