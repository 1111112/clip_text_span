## Imports
import sys
import os
os.chdir('/home/ic/Desktop/CogVideo')
project_root = os.path.abspath('')
sys.path.append(project_root)
import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
import cv2
import heapq
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import einops
from torchvision.datasets import ImageNet
from clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from clip_text_span.utils.visualization import image_grid, visualization_preprocess
from clip_text_span.prs_hook import hook_prs_logger
from matplotlib import pyplot as plt

def visualize_attention_maps(attention_map, lines, image_pil):
    """
    Visualize attention maps overlaid on the original image for two text prompts
    Args:
        attention_map: numpy array of attention weights
        lines: list of text prompts
        image_pil: PIL Image of the original image
    """
    # Convert PIL image to numpy array
    image_np = np.array(image_pil)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # First attention map
    ax1.imshow(image_np)  # Show original image
    attention1 = attention_map[0] - np.mean(attention_map, axis=0)
    attention1 = (attention1 - attention1.min()) / (attention1.max() - attention1.min())
    heatmap1 = ax1.imshow(attention1, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap1, ax=ax1)
    ax1.set_title(lines[0])
    ax1.axis('off')
    
    # Second attention map
    ax2.imshow(image_np)  # Show original image
    attention2 = attention_map[1] - np.mean(attention_map, axis=0)
    attention2 = (attention2 - attention2.min()) / (attention2.max() - attention2.min())
    heatmap2 = ax2.imshow(attention2, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap2, ax=ax2)
    ax2.set_title(lines[1])
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Show difference maps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Difference map (first - second)
    ax1.imshow(image_np)
    v = attention_map[0] - attention_map[1]
    v = (v - v.min()) / (v.max() - v.min())
    heatmap_diff1 = ax1.imshow(v, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap_diff1, ax=ax1)
    ax1.set_title(f'Difference: {lines[0]} - {lines[1]}')
    ax1.axis('off')
    
    # Difference map (second - first)
    ax2.imshow(image_np)
    v = attention_map[1] - attention_map[0]
    v = (v - v.min()) / (v.max() - v.min())
    heatmap_diff2 = ax2.imshow(v, cmap='jet', alpha=0.5)
    fig.colorbar(heatmap_diff2, ax=ax2)
    ax2.set_title(f'Difference: {lines[1]} - {lines[0]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_object_mask(attention_map, threshold=0.2):
    """
    Convert attention map to binary mask for the object region
    Args:
        attention_map: numpy array of attention weights
        threshold: float value for binarization
    Returns:
        binary_mask: numpy array of shape (H, W)
    """
    # Normalize attention map
    norm_map = attention_map - np.mean(attention_map, axis=0)
    norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min())
    
    # Binarize the map
    binary_mask = (norm_map > threshold).astype(np.uint8)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask


def visualize_object_mask(image_pil, attention_map, threshold=0.5, alpha=0.5):
    """
    Visualize object mask overlaid on the original image
    Args:
        image_pil: PIL Image
        attention_map: numpy array of attention weights
        threshold: float value for binarization
        alpha: float value for mask transparency
    """
    # Get binary mask
    binary_mask = get_object_mask(attention_map, threshold)
    
    # Convert PIL image to numpy array
    image_np = np.array(image_pil)
    
    # Create colored mask (red for visualization)
    colored_mask = np.zeros_like(image_np)
    colored_mask[binary_mask == 1] = [255, 0, 0]  # Red color for the mask
    
    # Create the overlay
    overlay = cv2.addWeighted(image_np, 1, colored_mask, alpha, 0)
    
    # Display results side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Binary mask
    ax2.imshow(binary_mask, cmap='gray')
    ax2.set_title('Binary Mask')
    ax2.axis('off')
    
    # Overlay
    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
## Hyperparameters


device = 'cuda:0'
pretrained = 'laion2b_s32b_b82k' # 'laion2b_s32b_b79k'
model_name = 'ViT-L-14' # 'ViT-H-14'
batch_size = 2 # only needed for the nn search
imagenet_path = '/datasets/ilsvrc_2024-01-04_1601/' # only needed for the nn search

## Loading Model

model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
model.to(device)
model.eval()
context_length = model.context_length
vocab_size = model.vocab_size
tokenizer = get_tokenizer(model_name)

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("Len of res:", len(model.visual.transformer.resblocks))

prs = hook_prs_logger(model, device)

## Load image

image_pil = Image.open('images/catdog.png')
image = preprocess(image_pil)[np.newaxis, :, :, :]
_ = plt.imshow(image_pil)


## Run the image:
prs.reinit()
with torch.no_grad():
    representation = model.encode_image(image.to(device), 
                                        attn_method='head', 
                                        normalize=False)
    attentions, mlps = prs.finalize(representation)  # attentions: [1, 32, 257, 16, 1024], mlps: [1, 33, 1024]

## Get the texts
lines = ['An image of a dog', 'An image of a cat']
texts = tokenizer(lines).to(device)  # tokenize
class_embeddings = model.encode_text(texts)
class_embedding = F.normalize(class_embeddings, dim=-1)

# visualization of attention map on top of the image
# Process attention maps
attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T

# Interpolate attention map to match image size
attention_map = F.interpolate(
    einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1),
    scale_factor=model.visual.patch_size[0],
    mode='bilinear'
).to(device)
attention_map = attention_map[0].detach().cpu().numpy()

# Visualize the overlaid attention maps
visualize_attention_maps(attention_map, lines, image_pil)
# Visualize object mask for dog (first attention map)
visualize_object_mask(image_pil, attention_map[0], threshold=0.23, alpha=0.4)