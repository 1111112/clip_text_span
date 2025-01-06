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
from torch.utils.data import DataLoader
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.visualization import image_grid, visualization_preprocess
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt
import os
import sys

# Add parent directory to path to make package imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
class ClipTextSpanDemo:
    def __init__(self):
        # Hyperparameters
        self.device = 'cuda:0'
        self.pretrained = 'laion2b_s32b_b82k'  # Alternative: 'laion2b_s32b_b79k'
        self.model_name = 'ViT-L-14'  # Alternative: 'ViT-H-14'
        self.batch_size = 2  # Only needed for nn search
        self.imagenet_path = '/datasets/ilsvrc_2024-01-04_1601/'  # Only needed for nn search

        # Initialize model and components
        self.setup_model()

    def setup_model(self):
        """Initialize the CLIP model, tokenizer and hooks"""
        self.model, _, self.preprocess = create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Store model parameters
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size
        self.tokenizer = get_tokenizer(self.model_name)
        
        # Initialize PRS logger hook
        self.prs = hook_prs_logger(self.model, self.device)
        
        # Print model info
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        print("Context length:", self.context_length)
        print("Vocab size:", self.vocab_size)
        print("Len of res:", len(self.model.visual.transformer.resblocks))

    # def process_image(self, image_path):
    #     """Load and process an image"""
    #     image_pil = Image.open(image_path)
    #     image = self.preprocess(image_pil)[np.newaxis, :, :, :]
    #     return image_pil, image
    def process_image(self, image_path):
        """Load and process an image"""
        # Print current working directory and script directory
        print("Current working directory:", os.getcwd())
        print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
        
        # Construct full path relative to script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at {full_path}")
        
        print(f"Attempting to load image from: {full_path}")
        image_pil = Image.open(full_path)
        image = self.preprocess(image_pil)[np.newaxis, :, :, :]
        return image_pil, image
    
    def get_image_representation(self, image):
        """Get image representation using the model"""
        self.prs.reinit()
        with torch.no_grad():
            representation = self.model.encode_image(
                image.to(self.device),
                attn_method='head',
                normalize=False
            )
            attentions, mlps = self.prs.finalize(representation)
        return attentions, mlps

    def process_texts(self, text_lines):
        """Process text prompts"""
        texts = self.tokenizer(text_lines).to(self.device)
        class_embeddings = self.model.encode_text(texts)
        return F.normalize(class_embeddings, dim=-1)

    def visualize_attention(self, attentions, class_embedding, text_lines):
        """Visualize attention maps for different text prompts"""
        attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
        
        # Resize attention map
        attention_map = F.interpolate(
            einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1),
            scale_factor=self.model.visual.patch_size[0],
            mode='bilinear'
        ).to(self.device)
        attention_map = attention_map[0].detach().cpu().numpy()

        # Visualize maps for each text prompt
        for idx, text in enumerate(text_lines):
            print(text)
            plt.figure()
            plt.imshow(attention_map[idx] - np.mean(attention_map, axis=0))
            
            # Calculate difference map
            v = attention_map[idx] - attention_map[1-idx]
            min_ = min((attention_map[0] - attention_map[1]).min(), 
                      (attention_map[1] - attention_map[0]).min())
            max_ = max((attention_map[0] - attention_map[1]).max(), 
                      (attention_map[1] - attention_map[1]).max())
            
            # Normalize and colorize
            v = v - min_
            v = np.uint8((v / (max_-min_))*255)
            high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            
            plt.colorbar()
            plt.axis('off')
            plt.show()

def main():
   
    demo = ClipTextSpanDemo()
    
    # Process image
    image_pil, image = demo.process_image('images/catdog.png')
    plt.imshow(image_pil)
    plt.show()
    
    # Get image representation
    attentions, mlps = demo.get_image_representation(image)
    
    # Process text prompts
    text_lines = ['An image of a dog', 'An image of a cat']
    class_embedding = demo.process_texts(text_lines)
    
    # Visualize attention maps
    demo.visualize_attention(attentions, class_embedding, text_lines)

if __name__ == "__main__":
    main()