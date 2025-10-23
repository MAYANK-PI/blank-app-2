import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dropout, Dense, LSTM, Embedding, Add,
    Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Segmentation + Captioning", layout="wide")
st.title("🖼️ Image Segmentation + Caption Generation (U-Net + CNN-LSTM)")

# ------------------------------
# 1. Load images from folder
# ------------------------------
IMAGE_DIR = "images"

if not os.path.exists(IMAGE_DIR):
    st.error(f"❌ Folder '{IMAGE_DIR}' not found! Please add it to your repo.")
    st.stop()

image_files = [
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not image_files:
    st.warning("⚠️ No images found in 'images/' folder.")
    st.stop()

st.success(f"📁 Found {len(image_files)} image(s) in '{IMAGE_DIR}'")

# ------------------------------
# 2. CNN Encoder (InceptionV3)
# ------------------------------
@st.cache_resource
def load_cnn_encoder():
    base = InceptionV3(weights="imagenet")
    model = Model(inputs=base.input, outputs=base.layers[-2].output)
    return model

cnn_encoder = load_cnn_encoder()

# ------------------------------
# 3. Caption Generator (mock)
# ------------------------------
def generate_caption(_model, _feature):
    # You can replace this with a trained captioning model
    return "A scenic view with nature and sky."

# ------------------------------
# 4. Build U-Net (for segmentation)
# ------------------------------
def build_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(3, 1, activation='softmax')(c5)

    model = Model(inputs, outputs)
    return model

unet_model = build_unet()

# ------------------------------
# 5. Display results
# ------------------------------
for idx, img_path in enumerate(image_files):
    st.subheader(f"🖼️ Image {idx + 1}: {os.path.basename(img_path)}")

    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((299, 299))
    x = np.expand_dims(kimage.img_to_array(img_resized), axis=0)
    x = preprocess_input(x)

    # CNN feature extraction
    feature = cnn_encoder.predict(x, verbose=0)
    caption = generate_caption(None, feature)

    # Segmentation
    img_small = img.resize((128, 128))
    img_arr = np.expand_dims(np.array(img_small) / 255.0, axis=0)
    mask = unet_model.predict(img_arr, verbose=0)[0]  # shape: (128,128,3)
    mask_classes = np.argmax(mask, axis=-1)  # class indices

    # Convert mask to color map
    colormap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # RGB colors for 3 classes
    mask_color = colormap[mask_classes]

    # Show images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask_color.astype(np.uint8), caption="Segmentation Mask (Color)", use_container_width=True)
    with col3:
        st.image(mask_classes, caption="Mask Class Indices", use_container_width=True)

    st.markdown(f"**📝 Generated Caption:** {caption}")
    st.divider()
