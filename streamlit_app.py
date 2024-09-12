import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

st.title("Real-Time Object Detection with YOLO")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.classNames = classNames

    def transform(self, frame):
        img = Image.fromarray(frame.to_ndarray(format="bgr24"))
        img_np = np.array(img)

        results = self.model(img_np)
        
        # Create a draw object for the image
        draw = ImageDraw.Draw(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
                label = f"{classNames[cls]} {confidence:.2f}"
                draw.text((x1, y1), label, fill="magenta")

        # Convert the image back to a numpy array
        return np.array(img)

webrtc_streamer(key="object-detection", video_transformer_factory=VideoTransformer)
