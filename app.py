import streamlit as st
import tempfile
import os
import shutil
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="YOLOv8 Video Detection", layout="centered")
st.title("Traffic Sign Detection with YOLOv8")

# Load your trained model (place 'best.pt' in the same folder)
@st.cache_resource
def load_model():
    model = YOLO("outputs/train_1/weights/best.pt")
    return model

model = load_model()

# Upload video
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name

    st.video(input_path)
    st.success("✅ Video uploaded successfully.")

    if st.button("🔍 Run Detection"):
        with st.spinner("Processing..."):

            # Create temp output directory
            output_dir = tempfile.mkdtemp()

            # Run YOLOv8 prediction
            model.predict(source=input_path, save=True, save_txt=False, project=output_dir, name='predict', conf=0.25)

            # Locate the predicted video
            pred_video_path = os.path.join(output_dir, "predict", os.path.basename(input_path))

            if os.path.exists(pred_video_path):
                st.success("✅ Detection completed!")
                st.video(pred_video_path)

                with open(pred_video_path, 'rb') as f:
                    st.download_button("⬇️ Download Result", f, file_name="detected_output.mp4", mime="video/mp4")
            else:
                st.error("Prediction video not found.")

        # Cleanup
        shutil.rmtree(output_dir)
