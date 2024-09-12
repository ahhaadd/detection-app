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
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0].item()
                cls = int(box.cls[0].item())

                img_np = self.draw_bounding_box(img_np, (x1, y1), (x2, y2), self.classNames[cls], confidence)

        return img_np

    def draw_bounding_box(self, img, start, end, label, confidence):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([start, end], outline="magenta", width=2)
        draw.text(start, f"{label} {confidence:.2f}", fill="magenta")
        return np.array(img_pil)

webrtc_streamer(key="object-detection", video_transformer_factory=VideoTransformer)
