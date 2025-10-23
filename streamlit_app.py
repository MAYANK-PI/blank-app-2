import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, Dense, LSTM, Embedding, Add,
Conv2D, MaxPooling2D, Conv2DTranspose, concatenate)
import matplotlib.pyplot as plt

----------------------------------------
STREAMLIT SETUP
----------------------------------------

st.set_page_config(page_title="Image Segmentation + Captioning", layout="wide")
st.title("🧠 Image Segmentation + Captioning (U-Net + CNN-LSTM)")
st.caption("Displays segmentation mask with indices + full-sentence caption generation")

----------------------------------------
1️⃣ Load Images from Folder
----------------------------------------

IMAGE_DIR = "images"

if not os.path.exists(IMAGE_DIR):
st.error(f"❌ Folder '{IMAGE_DIR}' not found. Please add it to your repo.")
st.stop()

image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
st.warning("⚠️ No images found in 'images/' folder.")
st.stop()

st.success(f"📁 Found {len(image_files)} image(s) in '{IMAGE_DIR}'")

----------------------------------------
2️⃣ CNN Encoder (InceptionV3)
----------------------------------------

@st.cache_resource
def load_cnn_encoder():
base = InceptionV3(weights="imagenet")
model = Model(inputs=base.input, outputs=base.layers[-2].output)
return model

cnn_encoder = load_cnn_encoder()

----------------------------------------
3️⃣ Caption Generator (Mock CNN-LSTM)
----------------------------------------

@st.cache_resource
def load_caption_model(vocab_size=5000, max_length=20):
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
