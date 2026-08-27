import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import pandas as pd
import importlib

import lpr_engine
from lpr_engine import LPREngine, correct_plate_syntax

# Streamlit Page Config
st.set_page_config(
    page_title="High-Accuracy License Plate Recognition (ANPR)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode / Vibrant Accent)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .result-text-box {
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: 3px;
        color: #10B981;
        background-color: #111827;
        padding: 0.85rem 1.25rem;
        border-radius: 0.5rem;
        border: 2px solid #059669;
        margin: 0.75rem 0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .badge-valid {
        background-color: #059669;
        color: white;
        padding: 0.35rem 0.65rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.95rem;
    }
    .badge-invalid {
        background-color: #D97706;
        color: white;
        padding: 0.35rem 0.65rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚗 High-Accuracy License Plate Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Detection with Deskewing, Preprocessing & Positional Syntax Correction</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

conf_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
ocr_engine = st.sidebar.selectbox("OCR Engine", ["easyocr", "tesseract_fallback"])
enable_deskew = st.sidebar.checkbox("Enable Deskew / Rotation Correction", value=True)
enable_syntax_correction = st.sidebar.checkbox("Enable Positional Syntax Correction", value=True)

if st.sidebar.button("🔄 Reload LPR Engine & Clear Cache"):
    st.cache_resource.clear()
    importlib.reload(lpr_engine)
    st.sidebar.success("Engine & Cache reloaded!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Format Verification Rules")
st.sidebar.info("Supported: Indian Standard (`MH12AB1234`), Bharat Series (`22BH1234AA`), & Universal International Plates (`JC 12 CG GP`)")

@st.cache_resource
def get_lpr_engine():
    return LPREngine()

engine = get_lpr_engine()

# Tabs Interface
tab1, tab2, tab3 = st.tabs(["📸 Image Recognition", "🎥 Video / Live Stream", "🧪 Syntax Corrector Lab"])

# ----------------- TAB 1: IMAGE RECOGNITION -----------------
with tab1:
    col_up, col_vis = st.columns([1, 1.2])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload a Vehicle Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="plate_uploader")
        
        if uploaded_file is not None:
            # Fix Streamlit BytesIO EOF read issue using seek(0)
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            image_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            
            if image_bgr is not None and image_bgr.size > 0:
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Uploaded Image", use_column_width=True)
                
                # Auto-trigger recognition or run on button click
                run_btn = st.button("🔍 Run License Plate Recognition", type="primary")
                
                if run_btn or "last_filename" not in st.session_state or st.session_state["last_filename"] != uploaded_file.name:
                    with st.spinner("Detecting plates and running multi-pass OCR..."):
                        start_time = time.time()
                        output = engine.process_image(image_bgr, conf_thresh=conf_threshold, ocr_engine=ocr_engine)
                        elapsed = time.time() - start_time
                        
                        st.session_state["output"] = output
                        st.session_state["elapsed"] = elapsed
                        st.session_state["last_filename"] = uploaded_file.name

    with col_vis:
        if "output" in st.session_state:
            output = st.session_state["output"]
            elapsed = st.session_state["elapsed"]
            vis_img = engine.draw_visualizations(output)
            
            st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption=f"Detection Results ({elapsed*1000:.1f} ms)", use_column_width=True)
            
            st.markdown("### 📋 Recognition Results")
            if not output["detections"]:
                st.warning("No license plates detected. Try adjusting the confidence threshold slider.")
            else:
                for idx, det in enumerate(output["detections"]):
                    plate_text = det["corrected_text"] or det["raw_text"] or "TEXT UNREADABLE"
                    with st.expander(f"Plate #{idx+1}: {plate_text}", expanded=True):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            if det["crop_enhanced"] is not None and det["crop_enhanced"].size > 0:
                                st.image(cv2.cvtColor(det["crop_enhanced"], cv2.COLOR_BGR2RGB), caption="Enhanced Crop", use_column_width=True)
                        with c2:
                            st.write(f"**Detector Conf:** `{det['det_conf']:.2f}`")
                            
                            st.markdown(f"**Predicted License Plate Text:**")
                            st.markdown(f'<div class="result-text-box">🔢 {plate_text}</div>', unsafe_allow_html=True)
                            
                            st.write(f"**Raw OCR Output:** `{det['raw_text']}`")
                            st.write(f"**Syntax Corrected:** `{det['corrected_text']}`")
                            
                            if det["is_valid"]:
                                st.markdown(f'<span class="badge-valid">VALID FORMAT: {det["format_type"]}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="badge-invalid">NON-STANDARD FORMAT</span>', unsafe_allow_html=True)

# ----------------- TAB 2: VIDEO RECOGNITION -----------------
with tab2:
    st.markdown("### 🎥 Video Analytics Pipeline")
    uploaded_video = st.file_uploader("Upload a Traffic Video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("▶️ Process Video Frames", type="primary"):
            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            log_container = st.empty()
            
            detections_log = []
            frame_count = 0
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= 150:
                    break
                
                if frame_count % 3 == 0:
                    output = engine.process_image(frame, conf_thresh=conf_threshold, ocr_engine=ocr_engine)
                    vis_frame = engine.draw_visualizations(output)
                    
                    st_frame.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    for d in output["detections"]:
                        detections_log.append({
                            "Frame": frame_count,
                            "Raw OCR": d["raw_text"],
                            "Corrected Text": d["corrected_text"],
                            "Confidence": round(d["det_conf"], 2),
                            "Format": d["format_type"]
                        })
                
                frame_count += 1
                progress_bar.progress(min(1.0, frame_count / total_frames))
                
            cap.release()
            st.success("Video processing complete!")
            
            if detections_log:
                df_log = pd.DataFrame(detections_log)
                st.dataframe(df_log)

# ----------------- TAB 3: SYNTAX CORRECTOR LAB -----------------
with tab3:
    st.markdown("### 🧪 Test Positional Syntax Correction Engine")
    st.markdown("Simulate OCR output errors (e.g. `MH1ZAB1Z34` $\\rightarrow$ `MH12AB1234`) and see how syntax mapping resolves them.")
    
    sample_inputs = ["MH1ZAB1Z34", "KAO5MBS678", "DL01CO1234", "22BH1234AA", "UP3ZAB9876", "JC 12 CG GP"]
    selected_sample = st.selectbox("Quick Preset Test Cases:", ["Custom Input..."] + sample_inputs)
    
    if selected_sample != "Custom Input...":
        test_text = selected_sample
    else:
        test_text = st.text_input("Enter Raw OCR String:", value="MH1ZAB1Z34")
        
    if test_text:
        corrected, is_valid, fmt_type = correct_plate_syntax(test_text)
        
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            st.metric("Raw OCR Input", test_text)
        with c_r2:
            st.metric("Syntax Corrected Output", corrected)
        with c_r3:
            st.metric("Format Validity", "VALID" if is_valid else "INVALID", delta=fmt_type)
