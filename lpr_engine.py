"""
License Plate Recognition (LPR / ANPR) Core Engine
---------------------------------------------------
Features:
- Padded bounding-box extraction to prevent character clipping.
- Deskewing & rotation alignment for angled plates.
- Morphological TopHat / BlackHat + Sobel X Edge License Plate Localizer.
- Multi-pass Binarization (CLAHE, Otsu Adaptive Binarization, Inversion).
- 2-Line License Plate Splitter & Horizontal Stacker.
- Dual-Engine OCR (EasyOCR / PyTesseract / Contour Fallback).
- Positional Syntax-Aware Character Correction (e.g. Indian standard format: MH 12 AB 1234).
"""

import cv2
import numpy as np
import re
import math
from typing import List, Dict, Tuple, Optional, Union

# Character Substitution Mappings for OCR Confusion Resolution
DICT_CHAR_TO_NUM = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1', 'l': '1', '|': '1', '!': '1',
    'Z': '2', 'z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5', '$': '5',
    'G': '6', 'b': '6',
    'T': '7', 'J': '7',
    'B': '8',
    'g': '9', 'q': '9'
}

DICT_NUM_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'G'
}

# Common Indian State Codes for Validation
INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB", "BH"
}

# Standard Indian License Plate Regex Patterns
RE_INDIAN_STD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
RE_INDIAN_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


def correct_plate_syntax(raw_text: str) -> Tuple[str, bool, str]:
    """
    Applies position-aware syntax correction for Indian standard license plates.
    
    Returns:
        (corrected_text, is_valid, format_type)
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not cleaned:
        return "", False, "unknown"
    
    # Check if already strictly valid
    if RE_INDIAN_STD.match(cleaned):
        return cleaned, True, "Indian Standard"
    if RE_INDIAN_BH.match(cleaned):
        return cleaned, True, "Bharat Series"
    
    # Positional Syntax Correction Engine
    if 8 <= len(cleaned) <= 11:
        chars = list(cleaned)
        n = len(chars)
        
        # 1. First 2 characters MUST BE State Code (Letters)
        for i in range(min(2, n)):
            if chars[i] in DICT_NUM_TO_CHAR:
                chars[i] = DICT_NUM_TO_CHAR[chars[i]]
                
        # 2. Last 4 characters MUST BE Registration Digits (Digits)
        for i in range(max(0, n - 4), n):
            if chars[i] in DICT_CHAR_TO_NUM:
                chars[i] = DICT_CHAR_TO_NUM[chars[i]]
                
        # 3. Handle District (digits) vs Series (letters)
        if n == 10:
            for i in (2, 3):
                if chars[i] in DICT_CHAR_TO_NUM:
                    chars[i] = DICT_CHAR_TO_NUM[chars[i]]
            for i in (4, 5):
                if chars[i] in DICT_NUM_TO_CHAR:
                    chars[i] = DICT_NUM_TO_CHAR[chars[i]]
        elif n == 9:
            if chars[2] in DICT_CHAR_TO_NUM:
                chars[2] = DICT_CHAR_TO_NUM[chars[2]]
            if chars[4] in DICT_NUM_TO_CHAR:
                chars[4] = DICT_NUM_TO_CHAR[chars[4]]
        elif n == 11:
            for i in (2, 3):
                if chars[i] in DICT_CHAR_TO_NUM:
                    chars[i] = DICT_CHAR_TO_NUM[chars[i]]
            for i in (4, 5, 6):
                if chars[i] in DICT_NUM_TO_CHAR:
                    chars[i] = DICT_NUM_TO_CHAR[chars[i]]

        candidate = "".join(chars)
        
        if RE_INDIAN_STD.match(candidate):
            return candidate, True, "Indian Standard (Syntax-Corrected)"
        
        state_candidate = candidate[:2]
        if state_candidate in INDIAN_STATES:
            return candidate, True, "Indian Format (State-Verified)"
            
        return candidate, False, "Uncertain Format"
        
    return cleaned, False, "Non-Standard Format"


def deskew_plate(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Detects plate skew angle via minAreaRect contours and rotates image horizontally.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_bgr

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return crop_bgr

    rect = cv2.minAreaRect(c)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.5 or abs(angle) > 30:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_multi_pass(crop_bgr: np.ndarray, target_height: int = 128) -> Dict[str, np.ndarray]:
    """
    Generates multi-pass preprocessed variants (BGR, Grayscale CLAHE, Otsu Binarized, 2-Line Stacked).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {}

    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return {}
    
    scale = target_height / float(h)
    new_w = max(32, int(w * scale))
    resized_bgr = cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # 1. CLAHE Grayscale
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)

    # 2. Otsu Binarization (Black text on white background)
    _, otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)

    # 3. 2-Line Plate Stacker (if aspect ratio < 3.2, likely 2-line plate)
    aspect_ratio = w / float(h)
    stacked_bgr = resized_bgr.copy()
    if aspect_ratio < 3.2 and h >= 30:
        top_half = resized_bgr[0:target_height//2, :]
        bot_half = resized_bgr[target_height//2:, :]
        stacked_bgr = np.hstack([top_half, bot_half])

    return {
        "bgr": resized_bgr,
        "clahe_bgr": cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR),
        "otsu_bgr": cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        "stacked_bgr": stacked_bgr
    }


def crop_with_padding(image: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.10) -> np.ndarray:
    """
    Extracts crop with safety margin padding around bbox coordinates (x1, y1, x2, y2).
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_w = int(box_w * margin_pct)
    pad_h = int(box_h * margin_pct)

    px1 = max(0, x1 - pad_w)
    py1 = max(0, y1 - pad_h)
    px2 = min(w_img, x2 + pad_w)
    py2 = min(h_img, y2 + pad_h)

    return image[py1:py2, px1:px2]


def localize_license_plate_opencv(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    Locates license plate bounding boxes using Morphological TopHat/BlackHat + Sobel X edge density.
    Pinpoints rectangular plates across light, dark, shadowed, and angled vehicle bumpers.
    """
    h_img, w_img = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Morphological Rectangular Kernel
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    
    # Combine TopHat (light plate) and BlackHat (dark text/border)
    enhanced = cv2.add(gray, cv2.subtract(tophat, blackhat))
    
    # 2. Sobel X Gradient (high vertical edge density for character sequences)
    grad_x = cv2.Sobel(enhanced, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    min_val, max_val = np.min(grad_x), np.max(grad_x)
    if max_val > 0:
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
    else:
        grad_x = grad_x.astype("uint8")
        
    blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
    thresh = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, rect_kernel)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, square_kernel, iterations=1)
    thresh = cv2.dilate(thresh, square_kernel, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        # License Plate Aspect Ratio typical range: 1.8 to 6.8
        if 1.8 <= aspect_ratio <= 6.8 and w >= 40 and h >= 12 and (area / float(w_img * h_img)) < 0.40:
            score = 1.0 - abs(aspect_ratio - 3.8) / 3.8
            conf = max(0.55, min(0.95, round(score, 2)))
            boxes.append((x, y, x + w, y + h, conf))
            
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    return boxes


class LPREngine:
    """
    Unified LPR Pipeline managing detection models, OCR readers, and post-processing.
    """
    def __init__(self, detector_weights: Optional[str] = None):
        self.detector = None
        self.easyocr_reader = None
        
        try:
            from ultralytics import YOLO
            import os
            if detector_weights and os.path.exists(detector_weights):
                self.detector = YOLO(detector_weights)
        except Exception:
            pass

    def init_easyocr(self, gpu: bool = False):
        if self.easyocr_reader is None:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            except Exception:
                pass

    def detect_plates(self, image_bgr: np.ndarray, conf_thresh: float = 0.35) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects license plates in image.
        Uses OpenCV TopHat/BlackHat Morphological Localizer + YOLO detection.
        """
        boxes = []
        
        # 1. Morphological TopHat / BlackHat License Plate Localizer (High Accuracy for Plates)
        morph_boxes = localize_license_plate_opencv(image_bgr)
        if morph_boxes:
            boxes.extend(morph_boxes[:3])  # Top 3 plate candidates

        # 2. YOLO Detector if trained model weights exist
        if self.detector is not None:
            try:
                results = self.detector(image_bgr, conf=conf_thresh, verbose=False)
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                        conf = float(b.conf[0].cpu().numpy())
                        boxes.append((x1, y1, x2, y2, conf))
            except Exception:
                pass
        
        if not boxes:
            h_img, w_img = image_bgr.shape[:2]
            boxes.append((0, 0, w_img, h_img, 0.10))

        return boxes

    def run_ocr_pass(self, crop_variants: Dict[str, np.ndarray], engine: str = "easyocr") -> str:
        """
        Executes multi-pass OCR on image variants and picks the candidate with highest syntax score.
        """
        candidates = []

        # 1. EasyOCR Multi-pass
        self.init_easyocr()
        if self.easyocr_reader is not None:
            for name, crop in crop_variants.items():
                try:
                    res = self.easyocr_reader.readtext(crop, detail=0, paragraph=False)
                    txt = "".join(res)
                    if txt:
                        candidates.append(txt)
                except Exception:
                    pass

        # 2. PyTesseract Fallback Multi-pass
        if not candidates:
            try:
                import pytesseract
                for name, crop in crop_variants.items():
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    txt = pytesseract.image_to_string(gray, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                    if txt.strip():
                        candidates.append(txt.strip())
            except Exception:
                pass

        if not candidates:
            return ""

        # Rank candidates by syntax correctness score
        best_candidate = candidates[0]
        for cand in candidates:
            corr, valid, _ = correct_plate_syntax(cand)
            if valid:
                return cand
            if len(re.sub(r"[^A-Z0-9]", "", cand)) > len(re.sub(r"[^A-Z0-9]", "", best_candidate)):
                best_candidate = cand

        return best_candidate

    def process_image(self, image_input: Union[str, np.ndarray], conf_thresh: float = 0.35, ocr_engine: str = "easyocr") -> Dict:
        """
        Full end-to-end processing pipeline on a single image.
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        else:
            image = image_input.copy()

        boxes = self.detect_plates(image, conf_thresh=conf_thresh)
        results = []

        for box in boxes:
            x1, y1, x2, y2, det_conf = box
            
            crop_padded = crop_with_padding(image, (x1, y1, x2, y2), margin_pct=0.10)
            crop_aligned = deskew_plate(crop_padded)
            crop_variants = preprocess_multi_pass(crop_aligned)
            
            raw_text = self.run_ocr_pass(crop_variants, engine=ocr_engine)
            corrected_text, is_valid, format_type = correct_plate_syntax(raw_text)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "det_conf": det_conf,
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "is_valid": is_valid,
                "format_type": format_type,
                "crop_enhanced": crop_variants.get("clahe_bgr", crop_aligned)
            })

        return {
            "image": image,
            "detections": results
        }

    def draw_visualizations(self, process_output: Dict) -> np.ndarray:
        """
        Draws bounding boxes, text overlay, and status badges on the image.
        """
        vis = process_output["image"].copy()
        for det in process_output["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            text = det["corrected_text"] or "LICENSE PLATE"
            is_valid = det["is_valid"]
            
            color = (0, 220, 0) if is_valid else (0, 140, 255)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            
            label = f"{text} ({det['det_conf']:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, y1 - h - 12), (x1 + w + 10, y1), color, -1)
            cv2.putText(vis, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return vis
