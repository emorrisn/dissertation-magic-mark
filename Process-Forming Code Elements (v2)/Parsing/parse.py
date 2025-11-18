from __future__ import annotations

from pathlib import Path
from typing import Iterable
from scripts.utils import (
	STUDENT_IMG_PATTERN,
	IMAGE_EXTS,
	DATA_DIR,
	ensure_dir,
)
from markschemes import MarkschemeProcessor

def process_images() -> None:
	# Placeholder for image processing logic
	print("Processing images...")
	# TODO: Implement the image -> text processor. 
	c = input(" ") # This is here to stop it looping like crazy


def main(argv: Iterable[str] | None = None) -> int:
	ensure_dir(DATA_DIR)
	has_files = True
	while has_files == True:
		if len(list(Path("data").glob("*"))) == 0:
			has_files = False
			print("No files found. Exiting.")
		else:
			print("Files found. Processing...")
			
			processor = MarkschemeProcessor()
			processor.process()

			process_images()
	return 0	

		


if __name__ == "__main__":
	raise SystemExit(main())

