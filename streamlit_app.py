import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image
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


'''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Perform object detection
    results = model(img)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Draw bounding boxes and labels
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return frame.from_ndarray(img, format="bgr24")

class VideoProcessor:
    def recv(self, frame):
        return process_frame(frame)

st.title("Real-Time Object Detection")

# Create the WebRTC stream
webrtc_ctx = st.components.v1.html(
    """
    <html>
    <body>
    <video id="video" autoplay></video>
    <script>
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
    });
    </script>
    </body>
    </html>
    """,
    height=480
)

# If webcam stream is available, process frames
if webrtc_ctx:
    st.write("Webcam is streaming.")
    webrtc_ctx = st.components.v1.html(
        """
        <html>
        <body>
        <video id="video" autoplay></video>
        <script>
        const video = document.getElementById('video');
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
        });
        </script>
        </body>
        </html>
        """,
        height=480
    )
else:
    st.write("Starting webcam...")
'''
