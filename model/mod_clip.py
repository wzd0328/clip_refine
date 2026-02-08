import os
import pdb
from copy import deepcopy
from functools import partial

import clip
import clip.model
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, OrderedDict, Optional
from dataclasses import dataclass

OPENCLIP_DATASET = {"ViT-H-14": "laion2b_s32b_b79k", "ViT-bigG-14": "laion2b_s39b_b160k"}

# Local model paths for offline loading
LOCAL_MODEL_PATHS = {
    "ViT-B/32": "/lpai/dataset/lhp/0-1-8/ clip/ViT-B-32.pt",
    "ViT-L/14": "/lpai/volumes/so-volume-bd-ga/lhp/models/openai_clip/ViT-L-14.pt",
}


def load_model(model_name: str):
    if model_name.startswith("Open_"):
        arch_name = model_name.split("_")[-1]
        model, _, preprocess = open_clip.create_model_and_transforms(arch_name, OPENCLIP_DATASET[arch_name])
        tokenizer = open_clip.get_tokenizer(arch_name)
    elif model_name == "siglip":
        model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    elif model_name == "dfn":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-H-14-378-quickgelu", pretrained="dfn5b")
        tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    else:  # Load CLIP models
        # Check if local model path exists
        if model_name in LOCAL_MODEL_PATHS and os.path.exists(LOCAL_MODEL_PATHS[model_name]):
            print(f"Loading CLIP model from local path: {LOCAL_MODEL_PATHS[model_name]}")
            model, preprocess = clip.load(LOCAL_MODEL_PATHS[model_name], device="cpu")
        else:
            model, preprocess = clip.load(model_name)
        tokenizer = partial(clip.tokenize, truncate=True)
    return model, preprocess, tokenizer

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    
class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MixClsHead(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):
        super().__init__()
        self.width = width
        
        mlp_width = int(width * mlp_ratio)
        self.mlps = nn.ModuleList([nn.Sequential(OrderedDict([
            ("c_norm", norm_layer(width)),
            ("c_fc", nn.Linear(width, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, width))
        ])) for _ in range(layers)])
        
        self.ln_mlp = norm_layer(width)
        self.text_projection = nn.Linear(width, output_dim)

        self.init_parameters()

    def init_parameters(self):
        proj_std = (self.width ** -0.5) * (2 ** -0.5)
        fc_std = (2 * self.width) ** -0.5

        for block in self.mlps:
            nn.init.normal_(block.c_fc.weight, std=fc_std)
            nn.init.normal_(block.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.width ** -0.5)
            self.text_projection.bias.data.fill_(0.0)

    def forward(self, x):

        for mlp in self.mlps:
            x = x + mlp(x)

        x = self.ln_mlp(x)
        x = self.text_projection(x)

        return x

@dataclass
class ClassHeadCfg:
    mlp_ratio: int = 4
    layers: int = 1
    vocab_size: int = 49408

def _build_cls_head(
        width,
        clshead_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    clshead_cfg = ClassHeadCfg(**clshead_cfg) if isinstance(clshead_cfg, dict) else clshead_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    head = MixClsHead(
        width=width,
        layers=clshead_cfg.layers,
        mlp_ratio=clshead_cfg.mlp_ratio,
        act_layer=act_layer,
        norm_layer=norm_layer,
        output_dim=clshead_cfg.vocab_size,
    )

    return head

class CLIP(torch.nn.Module):
    def __init__(
        self,
        backbone_name="ViT-B/32",
        feat_dim=2048,
        init_logit_scale=np.log(1 / 0.01),
        clshead_cfg=ClassHeadCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # Load Backbone Vision-Language Model
        self.backbone, self.img_preprocess, self.tokenizer = load_model(backbone_name)

        # freeze logit_scale
        self.backbone.logit_scale.requires_grad = False

        self.feat_dim = feat_dim
        self.convert_models_to_fp32()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)

        ## add superclass
        clshead_cfg = ClassHeadCfg(**clshead_cfg) if isinstance(clshead_cfg, dict) else clshead_cfg
        self.text_decoder = _build_cls_head(
            embed_dim=self.backbone.visual.output_dim,
            clshead_cfg=clshead_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        self.register_buffer("cap_fq", torch.zeros([1, clshead_cfg.vocab_size], dtype=torch.float64))
        self.register_buffer("num_samples", torch.zeros([1, 1], dtype=torch.float64))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)

    def convert_models_to_fp32(self):
        for p in self.parameters():
            p.data = p.data.float()

    def encode_image(self, x: torch.Tensor, normalized: bool = True):
        feat_v = self.backbone.encode_image(x)
        if normalized:
            feat_v = feat_v / feat_v.norm(dim=-1, keepdim=True)
        return feat_v

    def encode_text(self, t: torch.Tensor, normalized: bool = True):
        feat_t = self.backbone.encode_text(t)
        if normalized:
            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        return feat_t

    def forward(self, images: torch.Tensor, texts: torch.Tensor, image_embs=None, test=False):
        feat_v = self.backbone.encode_image(images)
        feat_t = self.backbone.encode_text(texts)

        if image_embs is None:
            image_embs = self.visual(images)

        logits = self.text_decoder(image_embs)
        labels = texts.clone()

        return {
            "image_features": feat_v,
            "text_features": feat_t,
            "logit_scale": self.logit_scale.exp(),
            "cap_fq": self.cap_fq,
            "num_samples": self.num_samples,
            "logits": logits,
            "labels": labels,
            "cls_logit_scale": torch.ones([1]),
        }

class ZeroshotClassifier(CLIP):
    def __init__(
        self,
        *args,
        pretrained_path,
        classname_file: str,
        class_prompt: str = "a photo of a",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        statedict = torch.load(pretrained_path, map_location="cpu")
        self.load_state_dict(statedict, strict=True)

        # Load class name dict for each label id
        self.id_classname_dict = self._make_classname_dict(classname_file)
        self.class_names = list(self.id_classname_dict.values())

        self.class_prompt = class_prompt
        self._classname_features = None

    def _make_classname_dict(self, metadata_path: str):
        id_classname_dict = {}
        with open(metadata_path, "r") as f:
            for line in f:
                assert len(line.split("\t")) == 2, "metadata must be composed of lines of <class id>\\t<classname>"
                cls_id, cls_name = line.split("\t")
                id_classname_dict[cls_id] = cls_name.replace("\n", "").replace("_", " ")
        return id_classname_dict

    def get_text_features(self, device) -> torch.Tensor:
        if self._classname_features is not None:
            return self._classname_features

        class_texts = torch.cat([self.tokenizer(f"{self.class_prompt} {c}") for c in self.class_names]).to(device)
        text_features = self.encode_text(class_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._classname_features = text_features
        return self._classname_features

    def forward(self, images, texts=None, test=False):
        # 1. Extract image features
        image_features = self.encode_image(images)

        # 2. Predict final labels
        class_similarities = (100.0 * image_features @ self.get_text_features(images.device).T).softmax(dim=-1)
        probs, preds = class_similarities.topk(1, dim=-1)
        modality_gap = F.mse_loss(image_features.mean(dim=0), self.get_text_features(images.device).mean(dim=0))
        output = {
            "preds": preds,
            "probs": probs,
            "feat_gap": modality_gap,
            "sparsity": 1.0,
        }
        return output

