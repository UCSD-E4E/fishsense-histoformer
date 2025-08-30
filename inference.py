import os
import cv2
import numpy as np
import torch
import argparse
import skimage.io

from utils import hist_match, align_to_four, npTOtensor
from Generator import Generator
from model_histoformer import Histoformer
from math import ceil

# Arguments
parser = argparse.ArgumentParser(description='Histoformer Inference Script')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to checkpoint directory')
parser.add_argument('--output_path', type=str, default='./results/output.png', help='Path to save output image')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of embedding features')

opt = parser.parse_args()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def array_hist_match(source, template, R_out, G_out, B_out):
    """
    Robust histogram matching for numpy arrays
    Returns:
        matched: matched image
        color_ratio: None (for compatibility)
    """
    matched = source.copy().astype(np.float32)
    
    # Process each channel
    for c, channel_out in enumerate([R_out, G_out, B_out]):
        # Compute source CDF
        hist, _ = np.histogram(source[..., c].flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        cdf = hist.cumsum()
        
        # Ensure target CDF is properly normalized
        channel_out = np.asarray(channel_out)
        target_cdf = channel_out.cumsum()
        target_cdf = target_cdf / target_cdf[-1]  # Normalize
        
        # Ensure equal length (256 bins)
        xp = np.linspace(0, 1, len(target_cdf))
        fp = np.linspace(0, 1, 256)
        target_cdf = np.interp(fp, xp, target_cdf)
        
        # Create mapping function
        mapping = np.interp(
            cdf, 
            target_cdf, 
            np.arange(256)
        )
        
        # Apply mapping
        matched[..., c] = np.clip(mapping[source[..., c].astype(np.uint8)], 0, 255)
    
    return matched.astype(np.uint8), None

# # Load image
# ori_img = skimage.io.imread(opt.image_path)
# if ori_img.ndim == 2:  # Grayscale image
#     ori_img = np.stack([ori_img]*3, axis=-1)
# ori_img = cv2.resize(ori_img, (288, 288))  # Resize to model input size

# # Compute histogram
# def compute_histogram(img):
#     hist = np.zeros((3, 256))
#     for i in range(3):
#         hist[i] = np.histogram(img[:,:,i], bins=256, range=(0, 256))[0]
#         hist[i] = hist[i] / hist[i].sum()
#     return torch.FloatTensor(hist).unsqueeze(0)  # Add batch dimension

# input_hist = compute_histogram(ori_img).to(device)

def load_image_with_alpha(image_path):
    """Load image and separate alpha channel if present"""
    # RGB
    img = skimage.io.imread(image_path)

    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        # RGBA image
        alpha = img[:, :, 3]
        img = img[:, :, :3]  # Discard alpha from RGB
    elif img.ndim == 2:
        # Grayscale image, convert to RGB
        img = np.stack([img]*3, axis=-1)

    return img, alpha

def pad_to_multiple(image, multiple=4):
    """Pad image to make dimensions divisible by `multiple` (for the generator)"""
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

def compute_histogram_at_fullres(img, window_size=288):
    """Compute sliding window histograms for full-res processing"""
    h, w = img.shape[:2]
    histograms = np.zeros((3, ceil(h/window_size), ceil(w/window_size), 256))
    
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            patch = img[i:i+window_size, j:j+window_size]
            for c in range(3):
                hist = np.histogram(patch[..., c], bins=256, range=(0, 255))[0]
                histograms[c, i//window_size, j//window_size] = hist / hist.sum()
    
    return torch.FloatTensor(histograms)

def process_in_patches(img, net_g, device, patch_size=512, overlap=32):
    h, w = img.shape[:2]
    stride = patch_size - overlap
    output = np.zeros_like(img, dtype=np.float32)
    weight = np.zeros_like(img, dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            y2 = min(y + patch_size, h)
            x1 = x
            x2 = min(x + patch_size, w)

            patch = img[y1:y2, x1:x2]
            padded_patch = pad_to_multiple(patch)
            tensor_patch = npTOtensor(padded_patch).to(device)

            with torch.no_grad():
                patch_out = net_g(tensor_patch)
                patch_out = patch_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                patch_out = patch_out[:y2-y1, :x2-x1]  # crop padding

            output[y1:y2, x1:x2] += patch_out
            weight[y1:y2, x1:x2] += 1.0

    output /= weight
    return (output * 255).clip(0, 255).astype(np.uint8)

# Load image at full resolution
ori_img, alpha = load_image_with_alpha(opt.image_path)
h, w = ori_img.shape[:2]

# Compute histograms using sliding windows
input_hist = compute_histogram_at_fullres(ori_img).to(device)

# Load models
model = Histoformer(embed_dim=opt.embed_dim, token_projection='linear', token_mlp='TwoDCFF').to(device)
net_g = Generator().to(device)

# Load weights
checkpoint_histo = torch.load(os.path.join(opt.checkpoint_dir, 'Histoformer-PQR_288_modifyloss.pth'), map_location=device)
checkpoint_netg = torch.load(os.path.join(opt.checkpoint_dir, 'Histoformer-PQR_netG_288_modifyloss.pth'), map_location=device)

model.load_state_dict(checkpoint_histo['state_dict'])
net_g.load_state_dict(checkpoint_netg['state_dict'])

model.eval()
net_g.eval()

# # Run inference
# with torch.no_grad():
#     # Pass histogram through Histoformer
#     pred_img = model(input_hist)

#     R_out = pred_img[:, 0].cpu().numpy()
#     G_out = pred_img[:, 1].cpu().numpy()
#     B_out = pred_img[:, 2].cpu().numpy()

#     # Modified histogram matching - work with arrays directly
#     def array_hist_match(source, template, R_out, G_out, B_out):
#         matched = source.copy()
#         for i, channel in enumerate([R_out, G_out, B_out]):
#             # Compute CDFs
#             hist, _ = np.histogram(source[..., i].flatten(), 256, [0, 256])
#             cdf = hist.cumsum()
#             cdf = cdf / cdf[-1]
            
#             template_cdf = channel.cumsum()
#             template_cdf = template_cdf / template_cdf[-1]
            
#             # Build mapping
#             mapping = np.interp(cdf, template_cdf, np.arange(256))
            
#             # Apply mapping
#             matched[..., i] = np.clip(mapping[source[..., i]], 0, 255)
        
#         return matched, None  # Returning None for the second value to match original function signature

#     RGB_hs_img, _ = array_hist_match(ori_img, ori_img, R_out, G_out, B_out)
#     RGB_hs_img = align_to_four(RGB_hs_img)
#     RGB_hs_img = npTOtensor(RGB_hs_img).to(device)

#     # Pass through Generator
#     img_gan = net_g(RGB_hs_img)
#     img_gan = img_gan.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     img_gan = np.clip(img_gan * 255, 0, 255).astype(np.uint8)

#     # Save output
#     os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
#     cv2.imwrite(opt.output_path, cv2.cvtColor(img_gan, cv2.COLOR_RGB2BGR))
#     print(f"Inference complete. Output saved to {opt.output_path}")

with torch.no_grad():
    # Process each window's histogram
    pred_img = model(input_hist.view(-1, 3, 256))  # Reshape for batch processing
    pred_img = pred_img.view(3, -1).cpu().numpy()  # Reshape back to full-res
    
    # Apply histogram matching to original image
    RGB_hs_img, _ = array_hist_match(ori_img, ori_img, 
                                    pred_img[0], pred_img[1], pred_img[2])
    
    # # Pad for generator (divisible by 4)
    # padded_img = pad_to_multiple(RGB_hs_img)
    # tensor_img = npTOtensor(padded_img).to(device)
    
    # # Process through generator
    # output = net_g(tensor_img)
    # output = output[0, :, :h, :w]  # Crop back to original dimensions
    
    # # Save result
    # output = output.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite(opt.output_path, (output * 255).clip(0, 255).astype(np.uint8)[..., ::-1])

    # Process through generator in memory-safe patches
    img_gan = process_in_patches(RGB_hs_img, net_g, device)

    # Save output
    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    cv2.imwrite(opt.output_path, img_gan[..., ::-1])