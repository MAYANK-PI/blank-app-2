import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, Embedding, Add, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
import matplotlib.pyplot as plt

# ----------------------------------------
# Streamlit UI setup
# ----------------------------------------
st.set_page_config(page_title="Image Segmentation + Captioning", layout="wide")
st.title("Image Segmentation + Caption Generation (U-Net + CNN-LSTM)")

# ----------------------------------------
# 1. Detect images folder
# ----------------------------------------
IMAGE_DIR = os.path.join(os.getcwd(), "images")

if not os.path.exists(IMAGE_DIR):
    st.error(f"Folder '{IMAGE_DIR}' not found! Please add an 'images' folder in your repository.")
    st.stop()

image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    st.warning("No images found in the 'images/' folder.")
    st.stop()

st.success(f"Found {len(image_files)} image(s) in '{IMAGE_DIR}'")

# ----------------------------------------
# 2. CNN Encoder (InceptionV3)
# ----------------------------------------
@st.cache_resource
def load_cnn_encoder():
    base = InceptionV3(weights="imagenet")
    model = Model(inputs=base.input, outputs=base.layers[-2].output)
    return model

cnn_encoder = load_cnn_encoder()

# ----------------------------------------
# 3. Caption Generator (placeholder CNN-LSTM)
# ----------------------------------------
@st.cache_resource
def load_caption_model(vocab_size=5000, max_length=20):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

caption_model = load_caption_model()

def generate_caption(model, feature):
    # Replace with your trained captioning logic
    return "A beautiful natural landscape with trees and sky."

# ----------------------------------------
# 4. Build simple U-Net
# ----------------------------------------
def build_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(3, 1, activation='softmax')(c5)  # 3 pseudo classes
    model = Model(inputs, outputs)
    return model

unet_model = build_unet()

# ----------------------------------------
# 5. Function: Apply random color map for classes
# ----------------------------------------
def apply_color_mask(mask):
    h, w, _ = mask.shape
    mask_class = np.argmax(mask, axis=-1)
    color_map = np.array([
        [255, 0, 0],    # Class 0 - Red
        [0, 255, 0],    # Class 1 - Green
        [0, 0, 255]     # Class 2 - Blue
    ])
    colored_mask = color_map[mask_class % 3].astype(np.uint8)
    return colored_mask

# ----------------------------------------
# 6. Process and Display Results
# ----------------------------------------
for idx, img_path in enumerate(image_files):
    st.subheader(f"Image {idx + 1}: {os.path.basename(img_path)}")

    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((299, 299))
    x = np.expand_dims(kimage.img_to_array(img_resized), axis=0)
    x = preprocess_input(x)

    # CNN Feature Extraction
    feature = cnn_encoder.predict(x, verbose=0)
    caption = generate_caption(caption_model, feature)

    # U-Net Segmentation
    img_small = img.resize((128, 128))
    img_arr = np.expand_dims(np.array(img_small) / 255.0, axis=0)
    mask_pred = unet_model.predict(img_arr, verbose=0)[0]
    colored_mask = apply_color_mask(mask_pred)

    # Resize mask to match image
    mask_resized = Image.fromarray(colored_mask).resize(img.size)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask_resized, caption="Segmentation Mask (Colored by Class)", use_container_width=True)

    st.write(f"**Generated Caption:** {caption}")
