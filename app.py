import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO('best.pt')  # Load your model file

# Load the model
model = load_model()

st.title("Solar Panel Detection")
st.write("Upload an image to detect solar panels")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","tif"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction on button click
    if st.button("Detect Solar Panels"):
        st.write("Running detection...")
        
        # Run inference
        results = model.predict(tmp_path, conf=0.25)
        
        # Get the result image with boxes
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display result
        st.image(result_img, caption="Detection Result", use_column_width=True)
        
        # Show metrics
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.write(f"Found {len(boxes)} solar panels")
            
            # Display confidence scores
            st.write("Confidence scores:")
            for i, conf in enumerate(boxes.conf):
                st.write(f"Panel {i+1}: {conf:.2f}")
        else:
            st.write("No solar panels detected")
    
    # Clean up the temporary file
    os.unlink(tmp_path)
