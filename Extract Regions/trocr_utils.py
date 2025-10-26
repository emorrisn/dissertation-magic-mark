from PIL import Image, ImageOps
import numpy as np
import cv2


def to_rgb_on_white(pil_img: Image.Image) -> Image.Image:
    """Composite any alpha onto white and return RGB PIL image."""
    img = pil_img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and 'transparency' in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "LA":
            img = img.convert("RGBA")
        elif img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1])
        return bg
    else:
        return img.convert("RGB")


def enhance_contrast_clahe(pil_img: Image.Image) -> Image.Image:
    """Apply CLAHE on the luminance channel (LAB) and return RGB PIL image."""
    # Convert PIL -> BGR numpy
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Convert back to PIL RGB
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img):
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def deskew_image(pil_img: Image.Image, max_angle=15) -> Image.Image:
    """Detect skew via minAreaRect on the binary image and rotate to deskew.

    Returns rotated PIL RGB image. If no skew detected or angle exceeds
    max_angle (safety), returns original image.
    """
    bgr = pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # binary inverse so text is white
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th == 0))
    if coords.shape[0] < 10:
        return pil_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # normalize angle
    if angle < -45:
        angle = angle + 90

    # only rotate if reasonable
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return pil_img

    # rotate using PIL to keep white background fill
    return pil_img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))


def remove_lines(pil_img: Image.Image, line_length_ratio: float = 0.5) -> Image.Image:
    """Detect long straight lines (like ruled lines) and remove them via inpainting."""
    bgr = pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=int(bgr.shape[1]*line_length_ratio), maxLineGap=20)
    if lines is None:
        return pil_img

    mask = np.zeros_like(gray)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)

    # dilate mask a bit
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return bgr_to_pil(inpainted)


def denoise_image(pil_img: Image.Image, h_color=10, h=10, templateWindowSize=7, searchWindowSize=21) -> Image.Image:
    """Apply Non-local Means denoising (color)."""
    bgr = pil_to_bgr(pil_img)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, h_color, h, templateWindowSize, searchWindowSize)
    return bgr_to_pil(den)


def adaptive_binarize(pil_img: Image.Image, block_size=15, c=10) -> Image.Image:
    """Adaptive Gaussian thresholding; returns a 3-channel RGB PIL image (black on white)."""
    bgr = pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # ensure block_size is odd and >=3
    if block_size % 2 == 0:
        block_size += 1
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return bgr_to_pil(rgb)


def morph_clean(pil_img: Image.Image, kernel_size=2, iterations=1) -> Image.Image:
    """Morphological opening to remove small noise and optionally closing to join strokes."""
    bgr = pil_to_bgr(pil_img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return bgr_to_pil(closed)


def unsharp_mask(pil_img: Image.Image, radius=1.0, amount=1.0) -> Image.Image:
    """Simple unsharp mask: sharpen image to emphasize strokes."""
    bgr = pil_to_bgr(pil_img)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
    return bgr_to_pil(sharpened)


def preprocess_for_trocr(pil_img: Image.Image,
                         pad_frac: float = 0.08,
                         min_height: int = 64,
                         upscale_threshold: int = 48,
                         do_deskew: bool = False,
                         do_remove_lines: bool = False,
                         do_denoise: bool = True,
                         do_binarize: bool = False,
                         do_morph: bool = True,
                         do_sharpen: bool = True) -> Image.Image:
    """Full preprocessing pipeline returning an RGB PIL image optimized for TrOCR.

    Optional steps can be toggled. Defaults aim for good general-purpose enhancement
    while keeping characters legible for TrOCR.
    """
    # Ensure RGB on white
    img = to_rgb_on_white(pil_img)

    # Optional deskew early (works better on grayscale/contrast)
    if do_deskew:
        try:
            img = deskew_image(img)
        except Exception:
            pass

    # Pad
    w, h = img.size
    pad = int(round(h * pad_frac))
    if pad > 0:
        img = ImageOps.expand(img, border=(pad, pad, pad, pad), fill=(255, 255, 255))

    # Contrast enhancement using CLAHE on L channel
    img = enhance_contrast_clahe(img)

    # Optional remove ruled lines
    if do_remove_lines:
        try:
            img = remove_lines(img)
        except Exception:
            pass

    # Optional denoise
    if do_denoise:
        try:
            img = denoise_image(img)
        except Exception:
            pass

    # Optional binarize (if you want strong black/white strokes)
    if do_binarize:
        try:
            img = adaptive_binarize(img)
        except Exception:
            pass

    # Morphological cleanup to remove small specks or connect strokes
    if do_morph:
        try:
            img = morph_clean(img, kernel_size=2, iterations=1)
        except Exception:
            pass

    # Sharpen slightly
    if do_sharpen:
        try:
            img = unsharp_mask(img, radius=1.0, amount=0.8)
        except Exception:
            pass

    # Upscale if too small
    w, h = img.size
    if h < min_height:
        scale = float(min_height) / float(h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    return img
