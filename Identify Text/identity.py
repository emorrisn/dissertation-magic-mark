import os
import time
import re
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import defaultdict

# Directories (robust, relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
regions_dir = os.path.join(SCRIPT_DIR, os.pardir, 'Extract Regions', 'result')
text_dir = os.path.join(SCRIPT_DIR, 'output')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load TrOCR model + processor
model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
print("TrOCR model loaded.")

# Ensure output directory exists
os.makedirs(text_dir, exist_ok=True)

def process_region_image(region_image_path):
    """Process a single region image with TrOCR"""
    try:
        # Open image
        image = Image.open(region_image_path).convert("RGB")

        # Preprocess
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Generate text with enhanced parameters for better quality
        # Using deterministic beam search for consistent, high-quality results
        generated_ids = model.generate(
            pixel_values,
            max_length=256,              # Allow longer text (default is often 64)
            num_beams=5,                 # Beam search with 5 beams (explores more possibilities)
            early_stopping=True,         # Stop when all beams reach end token
            do_sample=False,             # Deterministic decoding (no randomness)
            num_return_sequences=1,      # Return best sequence
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    except Exception as e:
        print(f"Failed to process region image {region_image_path}: {e}")
        return ""

def parse_filename(fname):
    """Extract img_no and line_no from filename like img_001_line01.jpg"""
    match = re.search(r"img_(\d+)_line(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Fallback for old format img_001.jpg
    match = re.search(r"img_(\d+)", fname)
    if match:
        return int(match.group(1)), 1  # Default to line 1
    return None, None

def process_document_folder(folder_name):
    """Process all region images in a document folder"""
    folder_path = os.path.join(regions_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        return False
    
    print(f"\nProcessing document: {folder_name}")
    
    # Get all region image FILES in the folder (ignore any subdirectories)
    region_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    if not region_files:
        print(f"  No region images found in {folder_name}")
        return False

    # Parse and sort region files by img_no
    files_with_meta = []
    for fname in region_files:
        img_no, line_no = parse_filename(fname)
        if img_no is not None:
            files_with_meta.append((fname, img_no, line_no))
    
    if not files_with_meta:
        print(f"  Could not parse any filenames in {folder_name}")
        return False
    
    # Sort by img_no to process in order
    files_with_meta.sort(key=lambda x: x[1])
    
    print(f"  Found {len(files_with_meta)} regions to process")
    
    # Dictionary to store text by line: {line_no: [(img_no, text), ...]}
    lines_dict = defaultdict(list)
    last_line_printed = 0
    
    # Process each region and organize by line
    for fname, img_no, line_no in files_with_meta:
        region_path = os.path.join(folder_path, fname)
        region_text = process_region_image(region_path)
        
        if region_text:
            lines_dict[line_no].append((img_no, region_text))
        
        # Check if we've moved to a new line - if so, print the completed previous line
        if line_no > last_line_printed:
            # Print all completed lines up to (but not including) current line
            for completed_line_no in range(last_line_printed + 1, line_no):
                if completed_line_no in lines_dict:
                    # Sort snippets by img_no and combine
                    snippets = sorted(lines_dict[completed_line_no], key=lambda x: x[0])
                    line_text = " ".join([text for _, text in snippets])
                    print(f"  LINE {completed_line_no}: {line_text}")
            last_line_printed = line_no - 1
    
    # Print any remaining completed lines
    max_line = max(lines_dict.keys()) if lines_dict else 0
    for completed_line_no in range(last_line_printed + 1, max_line + 1):
        if completed_line_no in lines_dict:
            snippets = sorted(lines_dict[completed_line_no], key=lambda x: x[0])
            line_text = " ".join([text for _, text in snippets]).replace(" .", " ")
            print(f"  LINE {completed_line_no}: {line_text}")
    
    if lines_dict:
        # Build final document by combining all lines in order
        output_lines = []
        for line_no in sorted(lines_dict.keys()):
            snippets = sorted(lines_dict[line_no], key=lambda x: x[0])
            line_text = " ".join([text for _, text in snippets if text]).replace(" .", " ")
            if line_text:
                output_lines.append(line_text)
        
        # Save the complete document text
        text_path = os.path.join(text_dir, folder_name + ".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        
        total_snippets = sum(len(snippets) for snippets in lines_dict.values())
        print(f"\n  Saved complete document to {text_path}")
        print(f"  Document contains {total_snippets} snippets across {len(output_lines)} lines\n")
        
        # Delete the processed folder
        import shutil
        shutil.rmtree(folder_path)
        print(f"  Deleted processed folder {folder_path}\n")
        
        return True
    else:
        print(f"  No text extracted from any regions in {folder_name}\n")
        return False

print("Watching for extracted text regions...")
print(f"Input: {regions_dir}")
print(f"Output: {text_dir}")
print("=" * 60)

while True:
    # List all document folders (each contains region images from one original document)
    if os.path.exists(regions_dir):
        document_folders = [
            f for f in os.listdir(regions_dir)
            if os.path.isdir(os.path.join(regions_dir, f))
        ]

        if document_folders:
            for folder_name in document_folders:
                print(f"Processing document folder: {folder_name}")
                success = process_document_folder(folder_name)
                if success:
                    print(f"Successfully processed document: {folder_name}")
                else:
                    print(f"Skipped document folder: {folder_name}")
        else:
            # Nothing new, wait before polling again
            time.sleep(2)
    else:
        print(f"Waiting for regions directory to be created: {regions_dir}")
        time.sleep(5)
