## TO RUN: py -3.12 run.py

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch
import os
from pathlib import Path
import logging

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

# Fix GPU memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_model():
    """Load the Nanonets OCR model and processor"""
    logging.info("Starting Nanonets-OCR2-3B model setup...")
    model_path = "nanonets/Nanonets-OCR2-3B"
    
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device selected: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load model with memory optimization
    if device == "cuda":
        logging.info("Loading model with float16 precision for memory efficiency...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.float16,  # Fixed: was torch_dtype, now dtype
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        logging.info("Loading model on CPU with float32 precision...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
    
    model.eval()
    logging.info("Model set to evaluation mode")
    
    logging.info("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    logging.info("Model loaded successfully!")
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory Allocated: {mem_allocated:.2f} GB")
        logging.info(f"GPU Memory Reserved: {mem_reserved:.2f} GB")
    
    return tokenizer, processor, model, device

def process_image(image_path, processor, model, max_new_tokens=2048):
    """Process a single image and extract text using Nanonets OCR"""
    logging.info(f"Processing image: {image_path}")
    
    # Clear GPU cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"GPU memory before processing: {mem_before:.2f} GB")
    
    # Simplified prompt for children's handwriting
    prompt = """Extract all text from the above document as if you were reading it naturally."""
    
    # Load image
    logging.info("Loading image...")
    image = Image.open(image_path)
    original_size = image.size
    logging.info(f"Original image size: {original_size}")
    
    # Resize image if too large to save memory
    max_image_size = 2048
    if max(image.size) > max_image_size:
        ratio = max_image_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logging.info(f"Resized image to: {image.size} (to reduce memory usage)")
    else:
        logging.info(f"Image size: {image.size}")
    
    # Create messages in the format expected by the model
    logging.info("Creating message format...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    
    # Process inputs
    logging.info("Applying chat template...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    logging.info("Processing image and text inputs...")
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    if torch.cuda.is_available():
        mem_after_input = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"GPU memory after input preparation: {mem_after_input:.2f} GB")
    
    # Generate text with memory-efficient settings
    logging.info(f"Starting text generation (max {max_new_tokens} tokens)...")
    logging.info("This may take a few minutes on GPU...")
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    
    logging.info("Generation complete! Decoding output...")
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    # Decode output
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Clear memory
    del inputs, output_ids, generated_ids, image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"GPU memory after cleanup: {mem_after:.2f} GB")
    
    logging.info("Processing complete!")
    return output_text[0]

def save_output(text, output_path):
    """Save extracted text to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    logging.info(f"Saved output to: {output_path}")

def main():
    logging.info("NANONETS OCR - Starting batch processing")
    
    # Setup model
    tokenizer, processor, model, device = setup_model()
    
    # Define paths
    data_dir = Path("./data")
    output_dir = Path("../Process Text/data")
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files from the data directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    logging.info(f"\nFound {len(image_files)} image(s) to process\n")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        logging.info(f"Processing image {idx}/{len(image_files)}")
        
        if not image_file.exists():
            logging.warning(f"File not found: {image_file}, skipping...")
            continue
        
        try:
            # Extract text (reduced tokens to save memory)
            # Increase max_new_tokens if you have longer documents
            text = process_image(str(image_file.absolute()), processor, model, max_new_tokens=2048)
            
            # Print to console
            print(f"Extracted text from {image_file.name}:")
            print(text)
            
            
            # Save to file
            output_file = output_dir / f"{image_file.stem}.txt"
            save_output(text, str(output_file))
            
            # Delete the processed image
            try:
                image_file.unlink()
                logging.info(f"Deleted processed image: {image_file.name}")
            except Exception as delete_error:
                logging.warning(f"Could not delete {image_file.name}: {str(delete_error)}")
            
        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    logging.info("All images processed successfully!")
    
if __name__ == "__main__":
    main()
