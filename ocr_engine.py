"""
ocr_engine.py
All OCR logic lives here, with no UI code, so it can be reused or tested alone.
"""
import io

import easyocr
import numpy as np
from PIL import Image, ImageDraw

# The Reader loads a neural network (slow, a few seconds), so we create it
# only once per language set and reuse it afterwards.
_readers = {}


def get_reader(languages=("en",), use_gpu=False):
    """Return a cached EasyOCR Reader for the given languages."""
    key = (tuple(languages), use_gpu)
    if key not in _readers:
        # On first run, EasyOCR downloads the model files (~100 MB).
        _readers[key] = easyocr.Reader(list(languages), gpu=use_gpu)
    return _readers[key]


def load_image(file_bytes):
    """Turn raw uploaded bytes into a clean RGB PIL image."""
    image = Image.open(io.BytesIO(file_bytes))
    return image.convert("RGB")  # drops transparency / palette modes


def extract_text(image, languages=("en",), use_gpu=False, min_confidence=0.0):
    """
    Run OCR on a PIL image.
    Returns a list of dicts: {"text": str, "confidence": float, "box": [[x,y]*4]}
    """
    reader = get_reader(languages, use_gpu)
    # EasyOCR accepts a numpy array. detail=1 -> returns box + text + confidence.
    results = reader.readtext(np.array(image), detail=1, paragraph=False)

    lines = []
    for box, text, conf in results:
        if conf >= min_confidence:
            lines.append({
                "text": text,
                "confidence": float(conf),
                "box": [[int(x), int(y)] for x, y in box],
            })
    return lines


def draw_boxes(image, lines):
    """Return a copy of the image with a rectangle around each detected line."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for line in lines:
        points = [tuple(p) for p in line["box"]]
        draw.polygon(points, outline="red", width=3)
    return annotated


def join_text(lines):
    """Combine all detected lines into one text block."""
    return "\n".join(line["text"] for line in lines)
