# Modified from:
# https://github.com/tsb0601/MMVP/blob/main/scripts/evaluate_vlm.py
# https://github.com/baaivision/DIVA/blob/main/run_DIVA_with_*.py

import os
import argparse
import csv
import torch
import clip
import open_clip
from tqdm import tqdm
from PIL import Image


def main(ckpt_path, benchmark_dir):
    # Load original model
    base_name = os.path.basename(ckpt_path)
    if base_name == 'openai_vit_l_14_224.ckpt':
        model, preprocess = clip.load("/lpai/volumes/so-volume-bd-ga/lhp/models/openai_clip/ViT-L-14.pt", device="cpu", jit=False)
        tokenize_func = clip.tokenize
        clip_fwd_func = clip_fwd

    elif base_name == 'openai_vit_l_14_336.ckpt':
        model, preprocess = clip.load(name="ViT-L/14@336px", device="cpu", jit=False)
        tokenize_func = clip.tokenize
        clip_fwd_func = clip_fwd

    elif base_name == 'openclip_vit_h_14_224.ckpt':
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", device="cpu", pretrained="laion2b_s32b_b79k")
        tokenize_func = open_clip.tokenize
        clip_fwd_func = openclip_fwd

    elif base_name == 'siglip_vit_so_14_384.ckpt':
        # following https://github.com/baaivision/DIVA?tab=readme-ov-file#pre-trained-weight-downloading
        local_dir = "./ViT-SO400M-14-SigLIP-384"
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="timm/ViT-SO400M-14-SigLIP-384", local_dir=local_dir, local_dir_use_symlinks=False)
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", device="cpu", pretrained=os.path.join(local_dir, "open_clip_pytorch_model.bin"),
                                                                     image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")
        tokenizer = open_clip.tokenizer.HFTokenizer(local_dir, context_length=64, clean="canonicalize")
        tokenize_func = lambda texts: tokenizer(texts, context_length=model.context_length)
        clip_fwd_func = openclip_fwd

    else:
        raise ValueError(f"Unsupported model type")

    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Eval
    print("Original performance:")
    results_original = benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir)
    print(results_original)

    # Load finetuned model
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.visual.load_state_dict(state_dict)

    # Eval
    print("\nAfter Model finetuning:")
    results_after = benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir)
    print(results_after)


# @torch.no_grad()
# def clip_fwd(model, imgs, text1, text2):
#     logits_per_image1, logits_per_text1 = model(imgs, text1)
#     logits_per_image2, logits_per_text2 = model(imgs, text2)
#     return logits_per_text1, logits_per_text2

@torch.no_grad()
def clip_fwd(model, imgs, text1, text2):
    feat_v1, feat_t1 = model.encode_image(imgs), model.encode_text(text1)
    logit_scale = model.logit_scale.exp()
    feat_v2, feat_t2 = model.encode_image(imgs), model.encode_text(text2)
    logits_per_img1 = feat_v1 @ feat_t1.T
    logits_per_img1 = logit_scale * logits_per_img1
    logits_per_text1 = logits_per_img1.T
    logits_per_img2 = feat_v2 @ feat_t2.T
    logits_per_img2 = logit_scale * logits_per_img2
    logits_per_text2 = logits_per_img2.T
    return logits_per_text1, logits_per_text2


@torch.no_grad()
def openclip_fwd(model, imgs, text1, text2):
    returned_tuple = model(imgs, text1) # len(returned_tuple) is 3 (4) for openclip (siglip)
    image_features = returned_tuple[0]
    text1_features = returned_tuple[1]
    returned_tuple = model(imgs, text2)
    image_features = returned_tuple[0]
    text2_features = returned_tuple[1]
    logits_per_image1 = 100.0 * image_features @ text1_features.T
    logits_per_text1 = logits_per_image1.T
    logits_per_image2 = 100.0 * image_features @ text2_features.T
    logits_per_text2 = logits_per_image2.T
    return logits_per_text1, logits_per_text2


@torch.no_grad()
def benchmark_model(preprocess, model, tokenize_func, clip_fwd_func, benchmark_dir, csv_outfile=None):
    device = next(model.parameters()).device

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    if csv_outfile is not None:
        csv_writer = csv.writer(csv_outfile)
        csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score']) # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features',
        'State and Condition', 'Quantity and Count',
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row

            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = tokenize_func([text1]).to(device)
            text2 = tokenize_func([text2]).to(device)

            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)

            logits_per_text1, logits_per_text2 = clip_fwd_func(model, imgs, text1, text2)
            probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
            probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]

            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            if csv_outfile is not None:
                csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        if csv_outfile is not None:
            csv_outfile.close()

    # Calculate percentage accuracies
    category_score_list = []
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        category_score_list.append(pair_accuracies[category])
    pair_accuracies['average_score'] = sum(category_score_list) / len(category_score_list)
    return pair_accuracies


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--un2clip_ckpt_path", type=str, default="./pretrained_models/openai_vit_l_14_224.ckpt",
#                         help="Path to un2CLIP finetuned model checkpoint")
#     parser.add_argument("--openaiclip_ckpt_path", type=str, default="./pretrained_models/openai_vit_l_14_224.ckpt", 
#                         help="Path to OpenAI CLIP finetuned model checkpoint")
#     parser.add_argument("--benchmark_dir", type=str, default="./datasets/MMVP_VLM",
#                         help="Path to MMVP_VLM benchmark dataset directory")
#     args = parser.parse_args()
#     print(args)
#     main(args.un2clip_ckpt_path, args.benchmark_dir)