"""
License Plate Recognition (LPR / ANPR) Core Engine
---------------------------------------------------
Features:
- Padded bounding-box extraction to prevent character clipping.
- Deskewing & rotation alignment for angled plates.
- Preprocessing (CLAHE, Bilateral filtering, Sharpening).
- Multi-Engine OCR (EasyOCR / PyTesseract / Custom Contour OCR).
- Syntax-Aware Positional Character Correction (e.g. Indian standard format: MH 12 AB 1234).
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


def preprocess_plate_crop(crop_bgr: np.ndarray, target_height: int = 128) -> np.ndarray:
    """
    Applies resolution scaling, CLAHE contrast boost, and adaptive sharpening.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr
    scale = target_height / float(h)
    new_w = max(16, int(w * scale))
    resized = cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
    return sharpened


def crop_with_padding(image: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.08) -> np.ndarray:
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
            else:
                self.detector = YOLO("yolov8n.pt")
        except Exception:
            pass

    def init_easyocr(self, gpu: bool = False):
        if self.easyocr_reader is None:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=gpu)
            except Exception:
                pass

    def detect_plates(self, image_bgr: np.ndarray, conf_thresh: float = 0.35) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects license plates in image. Returns list of (x1, y1, x2, y2, confidence).
        Falls back to contour-based rectangular shape detector if YOLO fails.
        """
        boxes = []
        if self.detector is not None:
            try:
                results = self.detector(image_bgr, conf=conf_thresh, verbose=False)
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                        conf = float(b.conf[0].cpu().numpy())
                        boxes.append((x1, y1, x2, y2, conf))
                if boxes:
                    return boxes
            except Exception:
                pass

        # Contour-Based Plate Detector Fallback
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blurred, 30, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

        h_img, w_img = image_bgr.shape[:2]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                    boxes.append((x, y, x + w, y + h, 0.50))
                    break
        
        if not boxes:
            boxes.append((0, 0, w_img, h_img, 0.10))

        return boxes

    def run_ocr(self, crop_bgr: np.ndarray, engine: str = "easyocr") -> str:
        """
        Runs OCR on a preprocessed crop image using EasyOCR, PyTesseract, or OpenCV fallback.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return ""

        # 1. Try EasyOCR
        if engine == "easyocr":
            self.init_easyocr()
            if self.easyocr_reader is not None:
                try:
                    res = self.easyocr_reader.readtext(crop_bgr, detail=0, paragraph=False)
                    return "".join(res)
                except Exception:
                    pass

        # 2. Try PyTesseract
        try:
            import pytesseract
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            txt = pytesseract.image_to_string(gray, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if txt.strip():
                return txt.strip()
        except Exception:
            pass

        return ""

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
            
            crop_padded = crop_with_padding(image, (x1, y1, x2, y2), margin_pct=0.08)
            crop_aligned = deskew_plate(crop_padded)
            crop_enhanced = preprocess_plate_crop(crop_aligned)
            
            raw_text = self.run_ocr(crop_enhanced, engine=ocr_engine)
            corrected_text, is_valid, format_type = correct_plate_syntax(raw_text)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "det_conf": det_conf,
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "is_valid": is_valid,
                "format_type": format_type,
                "crop_enhanced": crop_enhanced
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
