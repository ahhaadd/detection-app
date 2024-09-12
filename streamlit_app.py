import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

# Define a video processor class for webrtc
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.classNames = classNames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection
        results = self.model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Draw bounding boxes and labels
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f"{self.classNames[cls]} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Real-Time Object Detection")

# Create the WebRTC stream
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    video_frame_callback=None
)
