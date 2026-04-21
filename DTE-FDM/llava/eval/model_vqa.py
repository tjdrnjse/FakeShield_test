import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

import torchvision.models as models
import torch.nn as nn
from torchvision import transforms


CLASS_NAMES = ['AIGC inpainting', 'DeepFake', 'Photoshop']


class DomainTagGenerator:
    def __init__(self, model_path, num_classes=3, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """Run DTG inference with GradCAM.

        Returns:
            label (int): predicted class index (0=AIGC, 1=DeepFake, 2=Photoshop)
            probs (np.ndarray): softmax probabilities, shape (3,)
            cam (np.ndarray): GradCAM heatmap normalised to [0, 1], shape (224, 224)
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        _act = [None]
        _grad = [None]

        def fwd_hook(module, inp, out):
            _act[0] = out

        def bwd_hook(module, grad_in, grad_out):
            _grad[0] = grad_out[0]

        fwd_h = self.model.layer4.register_forward_hook(fwd_hook)
        bwd_h = self.model.layer4.register_full_backward_hook(bwd_hook)

        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        label = int(output.argmax(dim=1).item())

        output[0, label].backward()
        fwd_h.remove()
        bwd_h.remove()

        feat = _act[0]
        grad = _grad[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return label, probs, cam


def extract_clip_attention(image_path, model):
    """Extract last-layer CLIP ViT attention averaged over heads (CLS → patches).

    Returns:
        attn (np.ndarray): shape (H_patches, W_patches), normalised to [0, 1]
    """
    vision_tower = model.get_model().get_vision_tower()
    clip_model = vision_tower.vision_tower
    processor = vision_tower.image_processor

    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values']
    pixel_values = pixel_values.to(device=vision_tower.device, dtype=vision_tower.dtype)

    with torch.no_grad():
        outputs = clip_model(pixel_values, output_attentions=True)

    last_attn = outputs.attentions[-1]       # (1, heads, seq_len, seq_len)
    cls_attn = last_attn[0, :, 0, 1:]       # (heads, num_patches)
    cls_attn = cls_attn.mean(dim=0).cpu().float().numpy()

    n = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(n, n)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


def save_confidence_map(image_path, cam, probs, save_path, verdict, clip_attn=None):
    """Overlay GradCAM (and optionally CLIP attention) on the original image and save as JPEG."""
    orig = np.array(Image.open(image_path).convert('RGB'))
    orig_h, orig_w = orig.shape[:2]

    # DTG GradCAM overlay (JET)
    cam_full = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    dtg_overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 255).astype(np.uint8)

    prob_lines = [f'{n}: {p:.1%}' for n, p in zip(CLASS_NAMES, probs)]
    title_dtg = f'DTG GradCAM — Verdict: {verdict}\n' + '\n'.join(prob_lines)

    n_panels = 3 if clip_attn is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))

    axes[0].imshow(orig)
    axes[0].set_title('Original Image', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(dtg_overlay)
    axes[1].set_title(title_dtg, fontsize=11)
    axes[1].axis('off')
    sm_jet = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm_jet.set_array([])
    plt.colorbar(sm_jet, ax=axes[1], fraction=0.03, pad=0.02, label='DTG activation intensity')

    if clip_attn is not None:
        attn_full = cv2.resize(clip_attn, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_full), cv2.COLORMAP_INFERNO)
        attn_heatmap = cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB)
        clip_overlay = np.clip(0.55 * orig + 0.45 * attn_heatmap, 0, 255).astype(np.uint8)
        axes[2].imshow(clip_overlay)
        axes[2].set_title('LLaVA CLIP Attention\n(regions LLaVA focused on)', fontsize=11)
        axes[2].axis('off')
        sm_inf = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=1))
        sm_inf.set_array([])
        plt.colorbar(sm_inf, ax=axes[2], fraction=0.03, pad=0.02, label='CLIP attention intensity')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "llava-v1.5-13b"
    DTG = DomainTagGenerator(model_path=args.DTG_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    print(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Confidence maps saved next to the answers file
    conf_maps_dir = os.path.join(os.path.dirname(answers_file), 'confidence_maps')

    ans_file = open(answers_file, "w")
    print("======== DTE_FDM Detect Begin ========")
    for line in tqdm(questions):
        image_file = line["image"]
        qs = line["text"]

        # DTG: label + probabilities + GradCAM
        label, probs, cam = DTG.predict(image_file)

        if label == 0:
            qs = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + qs
        elif label == 1:
            qs = "This is a picture that is suspected to have been tampered with by DeepFake. " + qs
        elif label == 2:
            qs = "This is a picture that is suspected to have been tampered with by Photoshop. " + qs

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        full_image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(full_image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "image": image_file,
            "outputs": outputs,
            "dtg_probs": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
        }) + "\n")
        ans_file.flush()

        # Save GradCAM + CLIP attention map when image is judged as not tampered
        not_tampered = "has not been tampered with" in outputs.lower()
        if not_tampered:
            img_stem = os.path.splitext(os.path.basename(image_file))[0]
            conf_path = os.path.join(conf_maps_dir, f'{img_stem}_confidence.jpg')
            clip_attn = extract_clip_attention(full_image_path, model)
            save_confidence_map(full_image_path, cam, probs, conf_path, verdict='Not Tampered', clip_attn=clip_attn)

    ans_file.close()
    print("======== The detection result is saved to {} ========".format(args.answers_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--DTG-path", type=str, default="ckp/DTG.pth")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
