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
    # gt = torch.eye(y_t.shape[0], device=y_t.device, dtype=torch.long)
    # p_t = alpha * gt + (1.0 - alpha) * p_t
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
        lambda_dinoalign=0.0,
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
        self.lambda_dinoalign = lambda_dinoalign
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
        elif distill_loss == "specd":
            return self.specd_loss
            # return self.logitsspecd_loss
        elif distill_loss == "negkd":
            return self.neg_kd_loss
        elif distill_loss == "ancd":
            return self.ancd_loss
        elif distill_loss == "francd":
            # return self.francd_loss
            return self.francd_embed_loss

    def get_batch(self, batch, device=None, non_blocking=True):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    def clip_loss(self, feat_i, feat_t, logit_scale=1.0):
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logit_scale * logits_per_image, labels)
            + F.cross_entropy(logit_scale * logits_per_text, labels)
        ) / 2
        return total_loss

    def split_clip_loss(self, feat_i, feat_t):
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        B = logits_per_image.size(0)
        labels = torch.arange(B, device=logits_per_image.device)
        mask = ~torch.eye(B, device=logits_per_image.device, dtype=torch.bool)
        img_neg_logits = logits_per_image[mask]
        txt_neg_logits = logits_per_text[mask]
        img_scores = torch.clamp(img_neg_logits, min=0.0, max=1.0)
        txt_scores = torch.clamp(txt_neg_logits, min=0.0, max=1.0)
        img_dynamic_margins = 0.5 * torch.pow(img_scores, 3)
        txt_dynamic_margins = 0.5 * torch.pow(txt_scores, 3)
        logits_per_image[mask] += img_dynamic_margins
        logits_per_text[mask] += txt_dynamic_margins

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
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

    def specd_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            _, S_tea_i, _ = torch.linalg.svd(feat_it, full_matrices=False)
            
        _, S_stu_i, _ = torch.linalg.svd(feat_i, full_matrices=False)
        loss_kd = F.mse_loss(torch.log(S_stu_i + 1e-6), torch.log(S_tea_i + 1e-6))
        # diff = torch.log(S_tea_i + 1e-6) - torch.log(S_stu_i + 1e-6)
        # loss_kd = F.relu(diff).mean()
        return loss_kd

    def neg_kd_loss(self, images, texts, feat_i, feat_t):
        def get_negative_mask(batch_size):
            mask = torch.ones((batch_size, batch_size), dtype=torch.bool)
            mask.fill_diagonal_(0)
            return mask
        def distillation_on_negatives(student_logits, teacher_logits):
            batch_size = student_logits.shape[0]
            mask = get_negative_mask(batch_size).to(student_logits.device)
            # reshape 为 (B, B-1),把对角线元素剔除
            s_neg = student_logits[mask].view(batch_size, -1)
            t_neg = teacher_logits[mask].view(batch_size, -1)

            loss = distill(s_neg, t_neg.detach(), self.T)

            return loss
        with torch.no_grad():
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_per_image_t = feat_it @ feat_tt.T
            logits_per_text_t = feat_tt @ feat_it.T

        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T

        # labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        # loss_gt = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        loss_negkd = (distillation_on_negatives(logits_per_image, logits_per_image_t) +
                    distillation_on_negatives(logits_per_text, logits_per_text_t)) / 2

        # loss_kd = loss_gt + loss_negkd
        # return loss_kd
        # return loss_gt
        return loss_negkd

    def ancd_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it = out_t["image_features"]
            feat_it = F.normalize(feat_it, p=2, dim=-1)
        feat_i = F.normalize(feat_i, p=2, dim=-1)
        cos_sim = (feat_i * feat_it).sum(dim=-1)
        loss_kd = (1.0 - cos_sim).mean()
        return loss_kd
    
    def decompose_high_low_freq(self, logits, ratio=0.25):
        """
        辅助函数：利用 2D FFT 将 logits 分解为低频和高频部分
        :param logits: shape (B, B)
        :param ratio: 低频截断半径比例 (0.0 到 0.5 之间)
        """
        # 1. 转换到频域 (转为 float32 防止混合精度下报错)
        fft_logits = torch.fft.fft2(logits.float())
        fft_shift = torch.fft.fftshift(fft_logits)  # 将低频移到中心
        
        # 2. 生成低频掩码 (Mask)
        B1, B2 = logits.shape
        cy, cx = B1 // 2, B2 // 2
        Y, X = torch.meshgrid(torch.arange(B1), torch.arange(B2), indexing='ij')
        # 计算到中心的归一化距离
        dist = torch.sqrt(((Y - cy) / B1)**2 + ((X - cx) / B2)**2)
        mask = (dist <= ratio).to(logits.device).float()
        
        # 3. 施加掩码进行频域分离
        low_fft = fft_shift * mask
        high_fft = fft_shift * (1.0 - mask)
        
        # 4. 逆傅里叶变换回到空域 (取实部)
        logits_low = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
        logits_high = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real
        
        return logits_low, logits_high

    def francd_loss(self, images, texts, feat_i, feat_t):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_per_image_t = feat_it @ feat_tt.T
            logits_per_text_t = feat_tt @ feat_it.T
        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T
        img_low, img_high = self.decompose_high_low_freq(logits_per_image)
        img_t_low, img_t_high = self.decompose_high_low_freq(logits_per_image_t.detach())
        
        txt_low, txt_high = self.decompose_high_low_freq(logits_per_text)
        txt_t_low, txt_t_high = self.decompose_high_low_freq(logits_per_text_t.detach())

        loss_kd_low = (
            distill(img_low, img_t_low, self.T, self.alpha_blending) + 
            distill(txt_low, txt_t_low, self.T, self.alpha_blending)
        ) / 2
        loss_kd_high = (
            F.mse_loss(img_high, img_t_high) + 
            F.mse_loss(txt_high, txt_t_high)
        ) / 2

        lambda_high = 0.5 
        loss_kd = loss_kd_low + lambda_high * loss_kd_high
        
        return loss_kd

    def francd_embed_loss(self, images, texts, feat_i, feat_t, low_ratio=0.25):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            out_t = self.teacher(images, texts.squeeze())
            feat_it = out_t["image_features"]
        fft_s = torch.fft.rfft(feat_i.float(), dim=1)
        fft_t = torch.fft.rfft(feat_it.float(), dim=1)
        
        freq_dim = fft_s.shape[1]
        cutoff = int(freq_dim * low_ratio)
        
        low_s, low_t = fft_s[:, :cutoff], fft_t[:, :cutoff]
        loss_low = F.mse_loss(torch.view_as_real(low_s), torch.view_as_real(low_t))
        
        high_s, high_t = fft_s[:, cutoff:], fft_t[:, cutoff:]
        loss_high = F.mse_loss(torch.view_as_real(high_s), torch.view_as_real(high_t))
        
        lambda_low = 1.0
        lambda_high = 2.0
        
        loss_kd = lambda_low * loss_low + lambda_high * loss_high
        return loss_kd

    def dinov2_alignment_loss(self, feat_i, feat_v_dinov2):
        # print("clip img dim, dino dim", feat_i.shape[-1], feat_v_dinov2.shape[-1])
        feat_i = F.normalize(feat_i, p=2, dim=-1)
        feat_v_dinov2 = F.normalize(feat_v_dinov2, p=2, dim=-1)
        cos_sim = (feat_i * feat_v_dinov2).sum(dim=-1)
        loss_align = (1.0 - cos_sim).mean()
        return loss_align
    def uniformity_loss(self, x, t=2):
        x = F.normalize(x, p=2, dim=-1)
        sim_matrix = x @ x.T
        sq_pdist = 2.0 * (1.0 - sim_matrix)

        B = x.size(0)
        mask = torch.eye(B, device=x.device, dtype=torch.bool)
        off_diag_dist = sq_pdist[~mask].view(B, -1)

        return torch.log(torch.mean(torch.exp(-t * off_diag_dist)) + 1e-6)

    def center_loss(self, img_feat, txt_feat):
        # 计算 Batch 内的质心
        img_center = img_feat.mean(dim=0)
        txt_center = txt_feat.mean(dim=0)

        # 强制质心重合
        # 这能消除系统性的偏差，让正样本分数自然升高
        return F.mse_loss(img_center, txt_center)

    def covariance_regularization(self, x):
        """
        强制特征各维度去相关
        效果等同于让奇异值分布更均匀
        """
        B, D = x.shape
        x = x - x.mean(dim=0)

        # 计算协方差矩阵
        cov = (x.T @ x) / (B - 1)

        # 目标：协方差矩阵应该趋近于单位矩阵 I
        # 非对角线元素（相关性）应该为 0
        mask = ~torch.eye(D, device=x.device, dtype=torch.bool)
        off_diag_cov = cov[mask].pow(2).sum() / D

        # 对角线元素（方差）应该接近 1
        diag_cov = (torch.diag(cov) - 1).pow(2).mean()

        return off_diag_cov + diag_cov
    
    def spec_loss(self, feat_i):
        # 1. 计算奇异值
        _, S_i, _ = torch.linalg.svd(feat_i, full_matrices=False)
        
        # 2. 归一化奇异值，使其类似于概率分布
        S_i_prob = S_i / (S_i.sum() + 1e-8)
        
        # 3. 计算熵 (Shannon Entropy)
        # 均匀分布时熵最大：log(min(B, D))
        entropy_i = -(S_i_prob * torch.log(S_i_prob + 1e-8)).sum()
        
        # 4. 目标：最大化熵 -> 最小化（最大可能熵 - 当前熵）
        max_possible_entropy_i = torch.log(torch.tensor(float(len(S_i))))
        loss_i = (max_possible_entropy_i - entropy_i).pow(2)
        
        return loss_i

    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        images, texts = self.get_batch(batch, device=self.device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(images, texts.squeeze())
            feat_i, feat_t = out["image_features"], out["text_features"]
            # contrastive_loss = self.clip_loss(feat_i, feat_t, out["logit_scale"])
            contrastive_loss = self.clip_loss(feat_i, feat_t)
            # contrastive_loss = self.split_clip_loss(feat_i, feat_t)
            # uni_loss = self.uniformity_loss(feat_i)
            # center_loss = self.center_loss(feat_i, feat_t)
            # cov_loss = self.covariance_regularization(feat_i)
            # spec_loss = self.spec_loss(feat_i)
            loss_dinoalign = self.dinov2_alignment_loss(out["image_features_preproj"], out["image_features_dinov2"])
            total_loss = self.lambda_cont * contrastive_loss + self.lambda_dinoalign * loss_dinoalign

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
                # "loss_kd": loss_kd.detach().item(),
                # "loss_uni": uni_loss.detach().item(),
                # "loss_center": center_loss.detach().item(),
                # "loss_cov": cov_loss.detach().item(),
                # "loss_spec": spec_loss.detach().item(),
                "loss_dinoalign": loss_dinoalign.detach().item(),
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
        # lambda_struct=1.0,
        # lambda_mid=1.0,
        # lambda_dim=1.0,
        # lambda_ce=1.0,
        # lambda_negkd=1.0,
        regularization_decay=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.lambda_concat = lambda_concat
        # self.lambda_struct = lambda_struct
        # self.lambda_mid = lambda_mid
        # self.lambda_dim = lambda_dim
        # self.lambda_ce = lambda_ce
        # self.lambda_negkd = lambda_negkd
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

    # method4:全局结构相似 ---59.04
    # I-I_t ≈ T-T_t
    def struct_loss(self, images, texts, feat_i, feat_t, temperature=0.07):
        with torch.no_grad():
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_imgt = feat_it @ feat_it.T
            logits_txtt = feat_tt @ feat_tt.T

        # feat_i = F.normalize(feat_i, dim=-1)
        # feat_t = F.normalize(feat_t, dim=-1)

        logits_img = feat_i @ feat_i.T
        logits_txt = feat_t @ feat_t.T

        # I-I ≈ T_t-T_t
        # loss_kd = distill(logits_img, logits_txtt.detach(), self.T)

        kl_img_txt = distill(logits_img, logits_imgt, temperature)
        kl_txt_img = distill(logits_txt, logits_txtt, temperature)

        loss_kd = (kl_img_txt + kl_txt_img) / 2

        return loss_kd

    ## method5:全局结构相似+中间层特征
    # def struct_loss(self, images, texts, feat_i, feat_t, feat_mid_v=None, temperature=0.07):
    #     with torch.no_grad():
    #         self.teacher._register_visual_hooks(layer_index=-2)
    #         out_t = self.teacher(images, texts.squeeze())
    #         feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
    #         logits_imgt = feat_it @ feat_tt.T
    #         # logits_imgt = feat_it @ feat_it.T
    #         # logits_txtt = feat_tt @ feat_tt.T

    #         feat_mid_vt = out_t["feat_mid_v"]
    #         feat_mid_vt = F.normalize(feat_mid_vt, dim=-1) if feat_mid_vt is not None else None

    #     # feat_i = F.normalize(feat_i, dim=-1)
    #     # feat_t = F.normalize(feat_t, dim=-1)

    #     logits_img = feat_i @ feat_t.T
    #     # logits_img = feat_i @ feat_i.T
    #     # logits_txt = feat_t @ feat_t.T

    #     kl_img_txt = distill(logits_img, logits_imgt, temperature)
    #     # kl_txt_img = distill(logits_txt, logits_txtt, temperature)

    #     # loss_kd = (kl_img_txt + kl_txt_img) / 2
    #     loss_kd = kl_img_txt
    #     loss_mid = F.mse_loss(feat_mid_v, feat_mid_vt) if feat_mid_v is not None else 0.0

    #     return loss_kd, loss_mid

    def non_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def dim_loss(self, feat_i, feat_t, alpha=0.005):
        feat_i = self.model.module.bn(feat_i)
        feat_t = self.model.module.bn(feat_t)

        B = feat_i.shape[0]
        conv = torch.mm(feat_i.T, feat_t) / B

        diag = torch.diagonal(conv).add_(-1).pow(2).sum()
        non_diag = self.non_diagonal(conv).pow(2).sum()

        loss = diag + alpha * non_diag
        return loss

    ## method7: celoss + 负样本kd
    def neg_kd_loss(self, images, texts, feat_i, feat_t):
        def get_negative_mask(batch_size):
            mask = torch.ones((batch_size, batch_size), dtype=torch.bool)
            mask.fill_diagonal_(0)
            return mask
        def distillation_on_negatives(student_logits, teacher_logits):
            batch_size = student_logits.shape[0]
            mask = get_negative_mask(batch_size).to(student_logits.device)
            # reshape 为 (B, B-1)，把对角线元素剔除
            s_neg = student_logits[mask].view(batch_size, -1)
            t_neg = teacher_logits[mask].view(batch_size, -1)

            loss = distill(s_neg, t_neg.detach(), self.T)

            return loss
        with torch.no_grad():
            out_t = self.teacher(images, texts.squeeze())
            feat_it, feat_tt = out_t["image_features"], out_t["text_features"]
            logits_per_image_t = feat_it @ feat_tt.T
            logits_per_text_t = feat_tt @ feat_it.T

        logits_per_image = feat_i @ feat_t.T
        logits_per_text = feat_t @ feat_i.T

        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        loss_gt = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        # loss_negkd = 1e3 * (distillation_on_negatives(logits_per_image, logits_per_image_t) +
        #             distillation_on_negatives(logits_per_text, logits_per_text_t)) / 2

        # loss_kd = loss_gt + loss_negkd
        # return loss_kd
        return loss_gt
        # return loss_negkd

    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        self.update_decay_rate(engine.state.iteration)
        images, texts = self.get_batch(batch, device=self.device)

        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(images, texts.squeeze())
            feat_i, feat_t = out["image_features"], out["text_features"]
            ## student ce loss w/o logit_scale
            # contrastive_loss = self.clip_loss(feat_i, feat_t, out["logit_scale"])
            contrastive_loss = self.clip_loss(feat_i, feat_t)
            loss_main = self.lambda_cont * contrastive_loss

            ## student ce loss w/o logit_scale
            total_loss = loss_main

            ## method1
            # loss_concat = self.concat_loss(images, texts, feat_i, feat_t, out["feat_st_concat"])
            # loss_concat = self.concat_loss(images, texts, feat_i, out["concat_logit_scale"])
            # total_loss = loss_main + self.decay_rate * self.lambda_concat * loss_concat
            # total_loss = loss_main + self.lambda_concat * loss_concat

            ## method3
            # I-I ≈ T_t-T_t
            # loss_struct = self.struct_loss(images, texts, feat_i, feat_t)
            # total_loss = loss_main + self.decay_rate * self.lambda_struct * loss_struct

            ## method5
            # loss_struct, loss_mid = self.struct_loss(images, texts, feat_i, feat_t, out["feat_mid_v"])
            # total_loss = loss_main + self.lambda_struct * loss_struct + self.lambda_mid * loss_mid

            ## method6 dim loss
            # loss_dim = self.dim_loss(feat_i, feat_t)
            # total_loss = loss_main + self.lambda_dim * loss_dim

            ## method7
            # loss_negkd = self.neg_kd_loss(images, texts, feat_i, feat_t)
            # total_loss = loss_main + self.lambda_negkd * loss_negkd
            # loss_struct = self.struct_loss(images, texts, feat_i, feat_t)
            # total_loss = loss_main + self.lambda_ce * loss_ce + self.decay_rate * self.lambda_struct * loss_struct
            # loss_struct = self.struct_loss(images, texts, feat_i, feat_t)
            # total_loss = loss_main + self.lambda_negkd * loss_negkd + self.decay_rate * self.lambda_struct * loss_struct

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
                # "loss_struct": loss_struct.detach().item(),
                # "loss_mid": loss_mid.detach().item(),
                # "loss_dim": loss_dim.detach().item(),
                # "loss_ce": loss_ce.detach().item(),
                # "loss_negkd": loss_negkd.detach().item(),
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
