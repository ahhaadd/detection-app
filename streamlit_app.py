import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

st.title("Real-Time Object Detection with YOLO")

# Define a function to process video frames
def process_frame(frame):
    image = np.array(Image.open(io.BytesIO(frame)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Perform object detection
    results = model(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Draw bounding boxes and labels
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return image

# Function to display video
def display_video():
    st.write("Webcam is streaming...")
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
    
    return webrtc_ctx

# Display video stream
webrtc_ctx = display_video()

# If the webcam stream is available
if webrtc_ctx:
    st.write("Webcam is active. Processing frames...")
    while True:
        frame = st.camera_input("Frame")
        if frame:
            frame = process_frame(frame)
            # Convert the processed frame to PIL format and display it
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption="Processed Frame", use_column_width=True)

