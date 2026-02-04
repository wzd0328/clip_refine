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


class CLIP(torch.nn.Module):
    def __init__(
        self,
        backbone_name="ViT-B/32",
        feat_dim=2048,
        init_logit_scale=np.log(1 / 0.01),
        post_train_last_n_layers=2,
    ):
        super().__init__()
        # Load Backbone Vision-Language Model
        self.backbone, self.img_preprocess, self.tokenizer = load_model(backbone_name)

        # freeze logit_scale
        self.backbone.logit_scale.requires_grad = False

        self.feat_dim = feat_dim
        self.convert_models_to_fp32()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)

        self._freeze_backbone_but_last_n_layers(post_train_last_n_layers)

        self.visual_intermediate_features = {} 
        self._register_visual_hooks(layer_index=-2)

        # self.concat_logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)

        ## add fusion module ##
        ## method1
        # self.fusion_head = nn.Sequential(
        #         nn.Linear(512*2, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 512),
        #         # nn.Sigmoid()
        #     )

    def _freeze_backbone_but_last_n_layers(self, n):
        for p in self.parameters():
            p.requires_grad = False
            
        # for p in self.fusion_head.parameters():
        #     p.requires_grad = True
        self.logit_scale.requires_grad = True
        
        if hasattr(self.backbone, "visual"):
            visual_encoder = self.backbone.visual
            
            if hasattr(visual_encoder, "ln_post"):
                for p in visual_encoder.ln_post.parameters():
                    p.requires_grad = True
            if hasattr(visual_encoder, "ln_final"):
                 for p in visual_encoder.ln_final.parameters():
                    p.requires_grad = True

            if hasattr(visual_encoder, "transformer"):
                resblocks = visual_encoder.transformer.resblocks
            elif hasattr(visual_encoder, "trunk"): 
                resblocks = visual_encoder.trunk.blocks

            if n > 0 and len(resblocks) > 0:
                print(f"==> Unfreezing last {n} layers of Vision Encoder")
                layers_to_train = resblocks[-n:]
                for layer in layers_to_train:
                    for p in layer.parameters():
                        p.requires_grad = True
            else:
                print("==> Vision Encoder is fully frozen.")

    def _register_visual_hooks(self, layer_index=-2):
        def hook_fn(module, input, output):
            if output.shape[0] != input[0].shape[1]:
                 output = output.permute(1, 0, 2)
            self.visual_intermediate_features["feat"] = output

        if hasattr(self.backbone.visual, "transformer"):
            target_layers = self.backbone.visual.transformer.resblocks
        elif hasattr(self.backbone.visual, "trunk"):
            target_layers = self.backbone.visual.trunk.blocks
        else:
            return

        target_layers[layer_index].register_forward_hook(hook_fn)

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

    def encode_concat(self, feat_i: torch.Tensor, feat_t:torch.Tensor, normalized: bool = True):
        feat_st_concat = torch.cat([feat_i, feat_t], dim=1)
        # feat_st_concat = feat_i + feat_t
        feat_st_concat = self.fusion_head(feat_st_concat)
        feat_st_concat = F.normalize(feat_st_concat, dim=-1)

        # feat_ts_concat = torch.cat([feat_t, feat_i], dim=1)
        # feat_ts_concat = self.fusion_head(feat_ts_concat)
        # feat_ts_concat = F.normalize(feat_ts_concat, dim=-1)

        # return feat_st_concat, feat_ts_concat
        return feat_st_concat

    def forward(self, images: torch.Tensor, texts: torch.Tensor, test=False):
        feat_v = self.backbone.encode_image(images)
        feat_t = self.backbone.encode_text(texts)

        # active_mask = self.channel_mask(feat_v)
        # feat_v_masked = feat_v * active_mask
        # feat_v_masked = F.normalize(feat_v_masked, dim=-1)

        ## method3,4
        # return {"image_features": feat_v, "text_features": feat_t, "logit_scale": self.logit_scale.exp()}

        ## method2
        # return {"image_features": feat_v, "text_features": feat_t, "logit_scale": self.logit_scale.exp(), "concat_logit_scale": self.concat_logit_scale.exp()}

        ## method1
        # feat_st_concat = self.encode_concat(feat_v, feat_t)
        # return {"image_features": feat_v, "text_features": feat_t, "logit_scale": self.logit_scale.exp(), "feat_st_concat": feat_st_concat}

        ## method5 intermediate features
        feat_mid_v = self.visual_intermediate_features.get("feat", None)
        feat_mid_v = F.normalize(feat_mid_v, dim=-1) if feat_mid_v is not None else None
        return {"image_features": feat_v, "text_features": feat_t, "feat_mid_v": feat_mid_v, "logit_scale": self.logit_scale.exp()}


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

