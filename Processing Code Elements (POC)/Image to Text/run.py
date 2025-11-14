## TO RUN: py -3.12 run.py

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
import threading
import torch
import os
from pathlib import Path
import logging
import re

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

# Fix GPU memory fragmentation
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_model():
    """Load the Nanonets OCR model and processor"""
    logging.info("Starting Nanonets-OCR2-3B model setup...")
    model_path = "nanonets/Nanonets-OCR2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device selected: {device}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    logging.info("Model loaded successfully!")
    
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logging.info(f"GPU Memory Allocated: {mem_allocated:.2f} GB")
        logging.info(f"GPU Memory Reserved: {mem_reserved:.2f} GB")
    
    return processor, model, device

def process_image(image_path, processor, model, max_new_tokens=1024):
    """Process a single image and extract text using Nanonets OCR"""
    logging.info(f"Processing image: {image_path}")
    
    # Handwriting-focused prompt: ignore printed text, boilerplate, and blank regions
    prompt = (
        "Extract all handwritten text from the provided image. as if you were reading it naturally."
        "You are an OCR assistant. Transcribe ONLY handwritten content (letters, numbers, math, annotations). "
        "Ignore printed / typed text such as headers, instructions, watermarks, logos, page numbers, or form labels. "
        "Do NOT invent text. Do NOT output placeholder lines like '_' or '____'. Skip empty/blank lines. "
        "Preserve the natural order of the handwriting. If a line is illegible, omit it rather than guessing."
    )
    
    # Load image
    logging.info("Loading image...")
    image = Image.open(image_path)
    original_size = image.size
    logging.info(f"Original image size: {original_size}")
    
    # Resize image if too large to save memory
    max_image_size = 1536
    if max(image.size) > max_image_size:
        ratio = max_image_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logging.info(f"Resized image to: {image.size}")
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
    logging.info("Streaming tokens to console...")

    tokenizer = getattr(processor, "tokenizer", None)
   
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer; load a tokenizer or use a processor that bundles one.")
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        decode_kwargs={"clean_up_tokenization_spaces": True},
    )
    chunks = []

    def _consume():
        for piece in streamer:
            chunks.append(piece)
            print(piece, end="", flush=True)  # live stream to terminal

    t = threading.Thread(target=_consume)
    t.start()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            streamer=streamer,
        )
    t.join()
    logging.info("Generation finished.")
    output_text = ''.join(chunks)
 
    # Clear memory
    del inputs, output_ids, image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"GPU memory after cleanup: {mem_after:.2f} GB")
    
    logging.info("Processing complete!")
    return output_text

def save_output(text, output_path):
    """Save extracted text to file (ensures directory exists and logs absolute path)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    logging.info(f"Saved output to: {p.resolve()}")

def append_output(text, output_path):
    """Append extracted text to file (ensures directory exists and logs absolute path)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        if p.exists() and p.stat().st_size > 0:
            f.write("\n\n")  # separate pages with a blank line
        f.write(text)
    logging.info(f"Appended output to: {p.resolve()}")

def main():
    logging.info("NANONETS OCR - Starting batch processing")
    
    # Setup model
    processor, model, device = setup_model()
    
    # Define paths
    data_dir = Path("./data")
    eval_output_dir = Path("../Evaluate Work/data")
    markscheme_output_dir = Path("../Mark Scheme/data")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    markscheme_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluate Work outputs: {eval_output_dir.resolve()}")
    logging.info(f"Mark Scheme outputs:  {markscheme_output_dir.resolve()}")
    
    # Get all image files from the data directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    raw_image_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    # Patterns
    student_pattern = re.compile(r"^m(?P<session>\d+)_s(?P<student>\d+)_p(?P<page>\d+)$", re.IGNORECASE)
    markscheme_pattern = re.compile(r"^m(?P<session>\d+)_A(?P<artifact>\d+)$", re.IGNORECASE)

    # Classify and sort for deterministic processing
    student_images = []  # (sort_key, Path, session, student, page)
    markscheme_images = []  # (sort_key, Path, session, artifact)
    for image_file in raw_image_files:
        stem = image_file.stem
        m_mks = markscheme_pattern.match(stem)
        
        if m_mks:
            session_id = m_mks.group('session')
            artifact_id = m_mks.group('artifact')
            sort_key = (int(session_id), int(artifact_id), image_file.name.lower())
            markscheme_images.append((sort_key, image_file, session_id, artifact_id))
            continue

        m_std = student_pattern.match(stem)
        
        if m_std:
            session_id = m_std.group('session')
            student_id = m_std.group('student')
            page_id = m_std.group('page')
            sort_key = (int(session_id), int(student_id), int(page_id))
            student_images.append((sort_key, image_file, session_id, student_id, page_id))
            continue

        logging.warning("Filename '%s' did not match mark-scheme 'm{x}_A{y}' nor student 'm{x}_s{y}_p{z}'. Skipping...", image_file.name)

    student_images.sort(key=lambda item: item[0])
    markscheme_images.sort(key=lambda item: item[0])

    total_images = len(student_images) + len(markscheme_images)
    if total_images == 0:
        logging.info("No images matched known naming conventions. Nothing to process.")
        return

    logging.info(f"Found {total_images} image(s) to process ({len(markscheme_images)} mark-scheme, {len(student_images)} student)\n")

    # Process mark-scheme images first so downstream aggregator can pick them up quickly
    for idx, (_, image_file, session_id, artifact_id) in enumerate(markscheme_images, 1):
        logging.info(f"[MarkScheme] Processing image {idx}/{len(markscheme_images)}: {image_file.name}")
        if not image_file.exists():
            logging.warning(f"File not found: {image_file}, skipping...")
            continue
        try:
            text = process_image(str(image_file.absolute()), processor, model, max_new_tokens=1024)
            print(f"Extracted text from {image_file.name}:")
            print(text)

            # Output as artifact text back into Mark Scheme/data so that the Mark Scheme pipeline picks it up
            artifact_txt = markscheme_output_dir / f"m{session_id}_A{artifact_id}.txt"
            if artifact_txt.exists():
                append_output(text, str(artifact_txt))
            else:
                save_output(text, str(artifact_txt))

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

    # Process student images (original behavior)
    for idx, (_, image_file, session_id, student_id, page_id) in enumerate(student_images, 1):
        logging.info(f"[Student] Processing image {idx}/{len(student_images)}: {image_file.name}")
        if not image_file.exists():
            logging.warning(f"File not found: {image_file}, skipping...")
            continue
        try:
            text = process_image(str(image_file.absolute()), processor, model, max_new_tokens=1024)
            print(f"Extracted text from {image_file.name}:")
            print(text)

            student_file = eval_output_dir / f"m{session_id}_s{student_id}.txt"
            if student_file.exists():
                append_output(text, str(student_file))
            else:
                save_output(text, str(student_file))

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
    
    logging.info("All images processed and saved incrementally!")
    
if __name__ == "__main__":
    main()
