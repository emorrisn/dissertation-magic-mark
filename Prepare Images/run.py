# -*- coding: utf-8 -*-
"""
CRAFT Text Detection - Image Trimming Tool

Main entry point for preparing images by detecting and trimming text regions.
Place your input images in the 'data' folder and run this script.

Usage:
    python run.py                                    # Basic usage with defaults
    python run.py --margin 100                       # Increase trim margin
    python run.py --save_visualization True          # Save detection visualizations
    python run.py --refine                           # Enable link refiner for better accuracy
    python run.py --cuda False                       # Force CPU processing
"""

import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np

# Import supporting modules from scripts folder
from scripts import craft_utils
from scripts import imgproc
from scripts import file_utils
from scripts.craft import CRAFT
from scripts import trocr_utils

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection - Image Trimming Tool')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.5, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.45, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference (lower = faster, default: 1920)')
parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio (lower = faster, default: 1.0)')
parser.add_argument('--poly', default=True, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--folder', default='data', type=str, help='folder path to input images')
parser.add_argument('--output_folder', default='../Extract Text/data', type=str, help='folder path to save trimmed images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--margin', default=50, type=int, help='margin (in pixels) to add around detected text regions when trimming')
parser.add_argument('--save_visualization', default=False, type=str2bool, help='save visualization with detected regions (default: False)')
parser.add_argument('--cutter', default=False, type=str2bool, help='cut detected text regions into separate images (default: False, deprecated)')
parser.add_argument('--clear_input', default=True, type=str2bool, help='clear input data folder after processing (default: True)')

args = parser.parse_args()


""" For images in a folder """
image_list, _, _ = file_utils.get_files(args.folder)

# Create output folder for trimmed images
output_folder = args.output_folder
if not os.path.isdir(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# Optional: result folder for visualizations
result_folder = './result/'
if args.save_visualization and not os.path.isdir(result_folder):
    os.makedirs(result_folder, exist_ok=True)

def proc_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # Check if CUDA is actually available and adjust args.cuda accordingly
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU instead.")
        args.cuda = False
    elif args.cuda:
        print("CUDA is available - using GPU acceleration")
    
    # Performance optimization for CPU
    if not args.cuda:
        torch.set_num_threads(4)  # Limit CPU threads for better performance
        print("Running on CPU with 4 threads")
    
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True  # Enable cudnn autotuner for speed

    net.eval()
    
    # Set torch to inference mode for speed
    torch.set_grad_enabled(False)

    # LinkRefiner
    refine_net = None
    if args.refine:
        from scripts.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Processing image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = proc_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # Check if any text regions were detected
        if len(polys) == 0:
            print("  WARNING: No text regions detected in {}. Skipping trim.".format(os.path.basename(image_path)))
            continue

        # Calculate the bounding box that encompasses all detected text regions
        all_x_coords = []
        all_y_coords = []
        
        for poly in polys:
            pts = np.array(poly).reshape(-1, 2)
            all_x_coords.extend(pts[:, 0].tolist())
            all_y_coords.extend(pts[:, 1].tolist())
        
        # Find the outermost boundaries
        x_min = int(np.min(all_x_coords))
        x_max = int(np.max(all_x_coords))
        y_min = int(np.min(all_y_coords))
        y_max = int(np.max(all_y_coords))
        
        # Add margin and clip to image bounds
        h, w = image.shape[0], image.shape[1]
        x_min = max(0, x_min - args.margin)
        y_min = max(0, y_min - args.margin)
        x_max = min(w, x_max + args.margin)
        y_max = min(h, y_max + args.margin)
        
        # Crop the image to the content area
        trimmed_image = image[y_min:y_max, x_min:x_max].copy()
        
        # Save the trimmed image
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(output_folder, filename + file_ext)
        
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(output_path, trimmed_image[:,:,::-1])
        
        print("  Trimmed from {}x{} to {}x{} (reduced by {:.1f}%)".format(
            w, h, x_max-x_min, y_max-y_min,
            100 * (1 - ((x_max-x_min)*(y_max-y_min))/(w*h))
        ))
        
        # Optional: save visualization with detected regions
        if args.save_visualization:
            file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
            
            # Also save a visualization showing the trim area
            vis_image = image.copy()
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
            vis_path = os.path.join(result_folder, filename + "_trim_area.jpg")
            cv2.imwrite(vis_path, vis_image[:,:,::-1])

        # Optional: Legacy cutter functionality (deprecated, kept for compatibility)
        if args.cutter:
            # Ensure result subdirectory exists
            out_dir = os.path.join(result_folder, filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            # Compute bounding box centers for each polygon
            boxes_with_centers = []
            for i, poly in enumerate(polys):
                pts = np.array(poly).reshape(-1, 2)
                y_center = np.mean(pts[:, 1])
                x_center = np.mean(pts[:, 0])
                boxes_with_centers.append((i, x_center, y_center))

            # Group boxes into lines based on y-coordinate clustering
            # Sort by y-coordinate first
            boxes_with_centers.sort(key=lambda x: x[2])
            
            lines = []
            current_line = []
            line_threshold = 20  # pixels - adjust this if needed based on typical line spacing
            
            for box in boxes_with_centers:
                if not current_line:
                    current_line.append(box)
                else:
                    # Check if this box is on the same line as the current line
                    avg_y = np.mean([b[2] for b in current_line])
                    if abs(box[2] - avg_y) < line_threshold:
                        current_line.append(box)
                    else:
                        # Start a new line
                        lines.append(current_line)
                        current_line = [box]
            
            if current_line:
                lines.append(current_line)
            
            # Sort each line by x-coordinate (left to right) and flatten
            # Also track which line each box belongs to
            sort_keys = []
            for line_no, line in enumerate(lines, start=1):
                line.sort(key=lambda x: x[1])  # Sort by x_center
                # Add line number to each box tuple
                for box in line:
                    sort_keys.append((box[0], box[1], box[2], line_no))

            # Save crops in sorted order
            for out_idx, (orig_idx, _, _, line_no) in enumerate(sort_keys):
                poly = np.array(polys[orig_idx]).reshape(-1, 2).astype(int)
                # bounding rectangle for crop
                x_min = np.min(poly[:, 0])
                x_max = np.max(poly[:, 0])
                y_min = np.min(poly[:, 1])
                y_max = np.max(poly[:, 1])

                # Clip coordinates to image bounds
                h, w = image.shape[0], image.shape[1]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w - 1, x_max)
                y_max = min(h - 1, y_max)

                if x_max <= x_min or y_max <= y_min:
                    # skip invalid boxes
                    continue

                crop = image[y_min:y_max+1, x_min:x_max+1].copy()
                # Create a mask for the polygon region shifted to the crop origin
                pts_shifted = poly - np.array([x_min, y_min])
                pts_shifted = pts_shifted.reshape(-1, 2).astype(np.int32)
                mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
                try:
                    cv2.fillPoly(mask, [pts_shifted], 255)
                except Exception:
                    # fallback: fill bounding box if polygon drawing fails
                    mask[:, :] = 255

                try:
                    # Skip vertical crops (height > width) to keep line-level crops horizontal
                    ch, cw = crop.shape[0], crop.shape[1]
                    if ch > cw:
                        # log and continue
                        # Use out_idx in message but do not increment file numbering for skipped crops
                        print(f"    SKIP crop {orig_idx}: vertical crop (h={ch} > w={cw})")
                        continue

                    # Convert crop (numpy RGB) to PIL
                    pil_crop = Image.fromarray(crop)

                    # Preprocess for TrOCR: composite to white, pad, CLAHE, upscale
                    processed = trocr_utils.preprocess_for_trocr(pil_crop, pad_frac=0.08, min_height=64)

                    # Save as high-quality JPG for consistent downstream processing
                    save_path = os.path.join(out_dir, "img_{:03d}_line{:02d}.jpg".format(out_idx+1, line_no))
                    processed.save(save_path, format="JPEG", quality=95)
                except Exception as e:
                    print("Failed to save crop {}: {}".format(save_path, e))

    print("elapsed time : {}s".format(time.time() - t))
    
    # Clear input data folder if requested
    if args.clear_input:
        print(f"\nClearing input folder: {args.folder}")
        for filename in os.listdir(args.folder):
            file_path = os.path.join(args.folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"  Removed: {filename}")
            except Exception as e:
                print(f"  Failed to delete {filename}: {e}")
        print("Input folder cleared.")
