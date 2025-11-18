import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil

import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	pipeline,
)


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_system_prompt(system_prompt_path: Path) -> str:
	if system_prompt_path.exists():
		try:
			return system_prompt_path.read_text(encoding="utf-8").strip()
		except Exception:
			return ""
	return ""


def find_markschemes(data_dir: Path) -> list[tuple[str, Path]]:
	# Supports both patterns: m{n}_markscheme.txt and m{n}.markscheme.txt
	results: list[tuple[str, Path]] = []
	for p in data_dir.glob("*.txt"):
		name = p.name.lower()
		m = re.match(r"^(m\d+)[\._]markscheme\.txt$", name)
		if m:
			results.append((m.group(1), p))
	return sorted(results, key=lambda t: t[0])


def find_student_files(data_dir: Path, module_id: str) -> list[tuple[str, Path]]:
	# Pattern m{n}_s{X}.txt
	out = []
	for p in data_dir.glob(f"{module_id}_s*.txt"):
		name = p.name.lower()
		m = re.match(rf"^{module_id}_s(\d+)\.txt$", name)
		if m:
			out.append((m.group(1), p))
	# Sort numerically by session id
	return sorted(out, key=lambda t: int(t[0]))


def init_generator() -> tuple:
	if not torch.cuda.is_available():
		raise RuntimeError(
			"CUDA GPU not detected. Please install/use a CUDA-enabled PyTorch."
		)

	device_name = torch.cuda.get_device_name(0)
	print(f"Using device: {device_name}")

	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype="float16",
	)

	print("Loading model with 4-bit quantization (bitsandbytes)...")
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		quantization_config=bnb_config,
		device_map="auto",
	)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
	# Improve batching stability for causal models
	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"
	print("Model loaded!")

	gen = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
	)
	return gen, tokenizer


def clean_directory_contents(dir_path: Path) -> None:
	"""Remove all files and folders inside dir_path without deleting dir_path itself."""
	if not dir_path.exists():
		return
	for child in dir_path.iterdir():
		try:
			if child.is_dir():
				shutil.rmtree(child)
			else:
				child.unlink(missing_ok=True)
		except Exception as e:
			print(f"Warning: failed to remove {child}: {e}")


def build_messages(system_prompt: str, markscheme: str, submission: str) -> list[dict]:
	instructions = (
		"You are a strict but fair examiner. Evaluate the student's submission "
		"against the provided mark scheme. Return STRICT JSON with keys: "
		"marks_awarded (number), feedback (string), alert (boolean). Do not include extra commentary."
	)
	sys_content = (system_prompt + "\n\n" + instructions).strip()
	user_content = (
		f"Mark scheme:\n{markscheme}\n\nStudent submission:\n{submission}\n\n"
		"Return only JSON."
	)
	return [
		{"role": "system", "content": sys_content},
		{"role": "user", "content": user_content},
	]


def extract_json(text: str) -> dict | None:
	# Try to extract the outermost JSON object from the generation
	try:
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1 and end > start:
			candidate = text[start : end + 1]
			return json.loads(candidate)
	except Exception:
		return None
	return None


def process_student(
	generator,
	tokenizer,
	system_prompt: str,
	module_id: str,
	session_id: str,
	markscheme_text: str,
	submission_path: Path,
	output_dir: Path,
):
	submission_text = submission_path.read_text(encoding="utf-8", errors="ignore")
	messages = build_messages(system_prompt, markscheme_text, submission_text)

	prompt = tokenizer.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)

	outputs = generator(
		prompt,
		max_new_tokens=350,
		do_sample=True,
		temperature=0.3,
		top_p=0.9,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.eos_token_id,
		return_full_text=False,
	)

	gen_text = None
	if isinstance(outputs, list) and len(outputs):
		first = outputs[0]
		if isinstance(first, dict) and "generated_text" in first:
			gen_text = first["generated_text"]
		elif isinstance(first, list) and len(first) and isinstance(first[0], dict) and "generated_text" in first[0]:
			gen_text = first[0]["generated_text"]
	if gen_text is None:
		gen_text = str(outputs)

	parsed = extract_json(gen_text)
	result = {
		"module_id": module_id,
		"session_id": session_id,
		"student_file": submission_path.name,
		"model": MODEL_ID,
		"created_at": datetime.now(timezone.utc).isoformat(),
		"raw_text": gen_text,
		"marks_awarded": None,
		"feedback": None,
		"alert": False,
	}
	if isinstance(parsed, dict):
		result["marks_awarded"] = parsed.get("marks_awarded")
		result["feedback"] = parsed.get("feedback")
		# allow a variety of capitalizations and types
		alert_val = parsed.get("alert")
		if isinstance(alert_val, bool):
			result["alert"] = alert_val
		elif isinstance(alert_val, str):
			result["alert"] = alert_val.strip().lower() == "true"

	# Write to output
	out_path = output_dir / f"{module_id}_s{session_id}.json"
	out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Saved -> {out_path}")


def process_batch(
	generator,
	tokenizer,
	system_prompt: str,
	module_id: str,
	markscheme_text: str,
	batch_items: list[tuple[str, Path]],
	output_dir: Path,
):
	prompts: list[str] = []
	for session_id, submission_path in batch_items:
		submission_text = submission_path.read_text(encoding="utf-8", errors="ignore")
		messages = build_messages(system_prompt, markscheme_text, submission_text)
		prompt = tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		prompts.append(prompt)

	outputs = generator(
		prompts,
		batch_size=len(prompts),
		max_new_tokens=350,
		do_sample=True,
		temperature=0.3,
		top_p=0.9,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.eos_token_id,
		return_full_text=False,
	)

	# Normalize outputs to a flat list of generated strings aligned with prompts
	gen_texts: list[str] = []
	if isinstance(outputs, list) and len(outputs):
		if isinstance(outputs[0], list):
			for item in outputs:
				if item and isinstance(item[0], dict) and "generated_text" in item[0]:
					gen_texts.append(item[0]["generated_text"])
				else:
					gen_texts.append(str(item))
		elif isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
			gen_texts = [d.get("generated_text", "") for d in outputs]
		else:
			gen_texts = [str(x) for x in outputs]
	else:
		gen_texts = [str(outputs)] * len(prompts)

	for (session_id, submission_path), gen_text in zip(batch_items, gen_texts):
		parsed = extract_json(gen_text)
		result = {
			"module_id": module_id,
			"session_id": session_id,
			"student_file": submission_path.name,
			"model": MODEL_ID,
			"created_at": datetime.now(timezone.utc).isoformat(),
			"raw_text": gen_text,
			"marks_awarded": None,
			"feedback": None,
			"alert": False,
		}
		if isinstance(parsed, dict):
			result["marks_awarded"] = parsed.get("marks_awarded")
			result["feedback"] = parsed.get("feedback")
			alert_val = parsed.get("alert")
			if isinstance(alert_val, bool):
				result["alert"] = alert_val
			elif isinstance(alert_val, str):
				result["alert"] = alert_val.strip().lower() == "true"

		out_path = output_dir / f"{module_id}_s{session_id}.json"
		out_path.write_text(
			json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
		)
		print(f"Saved -> {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Evaluate student work against mark schemes using Llama 3.2 3B.")
	parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"), help="Input data directory containing mark schemes and student files.")
	parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"), help="Directory to write per-student results.")
	parser.add_argument("--system-prompt", default=str(Path(__file__).parent / "system_prompt.txt"), help="Path to system prompt file.")
	parser.add_argument("--delete-source", action="store_true", help="Delete processed student files after successful output.")
	parser.add_argument("--max-students", type=int, default=None, help="Optional cap on number of students processed per module.")
	parser.add_argument("--batch-size", type=int, default=2, help="Number of submissions to process concurrently on GPU (VRAM dependent).")
	# Clean output before run; support --no-clean-output to disable
	try:
		bool_opt = argparse.BooleanOptionalAction  # Python 3.9+
	except AttributeError:  # Fallback for older Python
		bool_opt = None
	if bool_opt is not None:
		parser.add_argument(
			"--clean-output",
			action=bool_opt,
			default=True,
			help="Clean the output directory before processing (default: True). Use --no-clean-output to disable.",
		)
	else:
		# Fall back to a pair of flags
		parser.add_argument("--clean-output", dest="clean_output", action="store_true", help="Clean output directory before processing.")
		parser.add_argument("--no-clean-output", dest="clean_output", action="store_false", help="Do not clean output directory before processing.")
		parser.set_defaults(clean_output=True)
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)
	system_prompt_path = Path(args.system_prompt)
	delete_source = bool(args.delete_source)
	clean_output = bool(getattr(args, "clean_output", True))

	if not data_dir.exists():
		print(f"Data directory not found: {data_dir}")
		sys.exit(1)

	marks = find_markschemes(data_dir)
	if not marks:
		print("No mark scheme files found (expected m{n}_markscheme.txt or m{n}.markscheme.txt). Aborting.")
		sys.exit(0)

	# Ensure output directory exists, then optionally clean its contents
	output_dir.mkdir(parents=True, exist_ok=True)
	if clean_output:
		print(f"Cleaning output directory: {output_dir}")
		clean_directory_contents(output_dir)

	system_prompt = load_system_prompt(system_prompt_path)
	generator, tokenizer = init_generator()

	for module_id, mark_path in marks:
		print(f"\n== Module {module_id} ==")
		mark_text = mark_path.read_text(encoding="utf-8", errors="ignore")
		students = find_student_files(data_dir, module_id)
		if not students:
			print(f"No student files for {module_id}. Skipping.")
			continue

		# Apply per-module limit before batching
		if args.max_students is not None:
			students = students[: args.max_students]

		batch_size = max(1, int(args.batch_size))
		for i in range(0, len(students), batch_size):
			batch_items = students[i : i + batch_size]
			process_batch(
				generator,
				tokenizer,
				system_prompt,
				module_id,
				mark_text,
				batch_items,
				output_dir,
			)
			if delete_source:
				for _session_id, sub_path in batch_items:
					try:
						sub_path.unlink(missing_ok=True)
					except Exception as e:
						print(f"Warning: failed to delete {sub_path}: {e}")

	print("\nAll done.")


if __name__ == "__main__":
	main()

