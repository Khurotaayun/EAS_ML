import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# Load model YOLO (pastikan 'best.pt' ada di folder yang sama)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

st.title("YOLOv5 Object Detection")
st.markdown("Upload gambar untuk deteksi menggunakan model kamu.")

# Upload file gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Diupload', use_column_width=True)

    # Deteksi dengan model YOLO
    with st.spinner('Sedang mendeteksi...'):
        results = model(image)
        # Visualisasi hasil deteksi
        results.render()  # menimpa results.imgs
        st.image(results.ims[0], caption='Hasil Deteksi', use_column_width=True)
