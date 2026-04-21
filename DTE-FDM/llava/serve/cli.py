import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from io import BytesIO
from transformers import TextStreamer
import json

import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import time


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
            probs (np.ndarray): softmax probabilities for each class, shape (3,)
            cam (np.ndarray): GradCAM heatmap normalised to [0, 1], shape (224, 224)
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Closures to capture layer4 activations and gradients for GradCAM
        _act = [None]
        _grad = [None]

        def fwd_hook(module, inp, out):
            _act[0] = out

        def bwd_hook(module, grad_in, grad_out):
            _grad[0] = grad_out[0]

        fwd_h = self.model.layer4.register_forward_hook(fwd_hook)
        bwd_h = self.model.layer4.register_full_backward_hook(bwd_hook)

        self.model.zero_grad()
        output = self.model(input_tensor)          # (1, 3)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        label = int(output.argmax(dim=1).item())

        # Backprop through the predicted class score to get GradCAM weights
        output[0, label].backward()

        fwd_h.remove()
        bwd_h.remove()

        feat = _act[0]   # (1, 2048, 7, 7)
        grad = _grad[0]  # (1, 2048, 7, 7)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().float().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

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

    # attentions: tuple(num_layers) of (1, num_heads, seq_len, seq_len)
    # seq_len = 1 CLS + num_patches
    last_attn = outputs.attentions[-1]       # (1, heads, seq_len, seq_len)
    cls_attn = last_attn[0, :, 0, 1:]       # (heads, num_patches)  — CLS row only
    cls_attn = cls_attn.mean(dim=0).cpu().float().numpy()  # (num_patches,)

    n = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(n, n)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


def save_confidence_map(image_path, cam, probs, save_path, verdict, clip_attn=None):
    """Overlay GradCAM heatmap (and optionally CLIP attention) on the original image.

    Args:
        image_path: path to the original input image
        cam: DTG GradCAM array shape (224, 224), values in [0, 1]
        probs: DTG softmax probabilities array shape (3,)
        save_path: output file path (.jpg)
        verdict: 'Tampered' or 'Not Tampered'
        clip_attn: optional CLIP ViT attention map shape (H_p, W_p), values in [0, 1]
    """
    orig = np.array(Image.open(image_path).convert('RGB'))
    orig_h, orig_w = orig.shape[:2]

    # DTG GradCAM overlay (JET colormap)
    cam_full = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    dtg_overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 255).astype(np.uint8)

    prob_lines = [f'{n}: {p:.1%}' for n, p in zip(CLASS_NAMES, probs)]
    prob_text = '\n'.join(prob_lines)
    title_dtg = f'DTG GradCAM — Verdict: {verdict}\n{prob_text}'

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
        # CLIP attention overlay (INFERNO colormap — visually distinct from JET)
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
    print(f"======== Confidence map saved to {save_path} ========")


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def DTE_FDM_init(args):
    # Model
    disable_torch_init()
    model_name = "llava-v1.5-13b"
    DTG = DomainTagGenerator(model_path=args.DTG_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    return tokenizer, model, image_processor, context_len, DTG, model_name

def DTE_FDM_cli(args):
    print("======== DTE_FDM Model Loading ========")
    tokenizer, model, image_processor, context_len, DTG, model_name = DTE_FDM_init(args)
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image_path = args.image_path
    image = load_image(image_path)

    # DTG: get label, class probabilities, and GradCAM heatmap
    label, probs, cam = DTG.predict(image_path)
    print("======== DTE_FDM Model Loaded ========")

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying special attention to the edges of the objects, capturing some unnatural edges and perspective relationships, some incorrect semantics, unnatural lighting and darkness etc."
    if label == 0:
        inp = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + inp
    elif label == 1:
        inp = "This is a picture that is suspected to have been tampered with by DeepFake. " + inp
    elif label == 2:
        inp = "This is a picture that is suspected to have been tampered with by Photoshop. " + inp

    print(f"{roles[1]}: ", end="")

    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("======== DTE_FDM Detect Begin ========")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    outputs = outputs.replace("<s>","").replace("</s>","")

    # Save detection result JSON (now includes DTG probabilities)
    with open(args.output_path, "w") as f:
        json.dump({
            "image": image_path,
            "outputs": outputs,
            "dtg_probs": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
        }, f)

    print("======== The detection result is saved to {} ========".format(args.output_path))

    # Save GradCAM + CLIP attention map when image is judged as not tampered
    not_tampered = "has not been tampered with" in outputs.lower()
    if not_tampered:
        base = os.path.splitext(args.output_path)[0]
        conf_path = base + '_confidence.jpg'
        clip_attn = extract_clip_attention(image_path, model)
        save_confidence_map(image_path, cam, probs, conf_path, verdict='Not Tampered', clip_attn=clip_attn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--DTG-path", type=str, default="ckp/DTG.pth")
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DTE_FDM_cli(args)
