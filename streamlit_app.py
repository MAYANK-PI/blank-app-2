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
# 4. Build improved U-Net for better visualization
# ----------------------------------------
def build_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
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

    # Change to sigmoid for binary-like output or keep softmax with fewer classes
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)  # Single channel for binary mask
    model = Model(inputs, outputs)
    return model

unet_model = build_unet()

# ----------------------------------------
# 5. Improved mask visualization functions
# ----------------------------------------
def apply_semantic_colormap(mask):
    """Apply a meaningful colormap for semantic segmentation"""
    # For single channel mask (sigmoid output)
    if len(mask.shape) == 2 or mask.shape[2] == 1:
        mask = mask.squeeze()
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Define colors for different intensity levels
        low_mask = mask < 0.33
        medium_mask = (mask >= 0.33) & (mask < 0.66)
        high_mask = mask >= 0.66
        
        # Blue for low, Green for medium, Red for high
        colored_mask[low_mask] = [0, 0, 255]    # Blue
        colored_mask[medium_mask] = [0, 255, 0] # Green
        colored_mask[high_mask] = [255, 0, 0]   # Red
        
    else:
        # For multi-class (softmax output)
        mask_class = np.argmax(mask, axis=-1)
        colored_mask = np.zeros((*mask_class.shape, 3), dtype=np.uint8)
        
        # Meaningful colors for different classes
        class_colors = {
            0: [0, 0, 0],       # Background - Black
            1: [0, 255, 0],     # Class 1 - Green (e.g., vegetation)
            2: [0, 0, 255],     # Class 2 - Blue (e.g., water/sky)
            3: [255, 0, 0],     # Class 3 - Red (e.g., buildings)
            4: [255, 255, 0],   # Class 4 - Yellow
            5: [255, 0, 255],   # Class 5 - Magenta
        }
        
        for class_id, color in class_colors.items():
            colored_mask[mask_class == class_id] = color
    
    return colored_mask

def create_overlay(image, mask, alpha=0.5):
    """Overlay mask on original image"""
    # Ensure both are the same size
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # Blend image and mask
    overlay = (img_array * (1 - alpha) + mask_array * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

def create_contour_overlay(image, mask, color=[255, 0, 0], thickness=2):
    """Create contour lines on original image"""
    from PIL import ImageDraw
    
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Convert mask to binary for contours
    mask_array = np.array(mask.convert('L')) > 128
    
    # Find contours (simple approach)
    contours = []
    h, w = mask_array.shape
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask_array[i, j]:
                # Check if this pixel is on boundary
                neighbors = [
                    mask_array[i-1, j], mask_array[i+1, j],
                    mask_array[i, j-1], mask_array[i, j+1]
                ]
                if not all(neighbors):
                    contours.append((j, i))
    
    # Draw contours
    for x, y in contours:
        draw.rectangle([x, y, x+thickness, y+thickness], fill=tuple(color), outline=tuple(color))
    
    return overlay

# ----------------------------------------
# 6. Process and Display Results
# ----------------------------------------

# Add visualization options
st.sidebar.header("Visualization Options")
viz_mode = st.sidebar.selectbox(
    "Choose visualization mode:",
    ["Colored Mask", "Overlay", "Contour Overlay", "Side by Side"]
)

alpha = st.sidebar.slider("Overlay transparency", 0.1, 0.9, 0.5) if viz_mode == "Overlay" else 0.5

for idx, img_path in enumerate(image_files):
    st.subheader(f"Image {idx + 1}: {os.path.basename(img_path)}")

    # Load and process image
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
    
    # Process mask based on output type
    if mask_pred.shape[-1] == 1:  # Sigmoid output
        mask_processed = (mask_pred.squeeze() * 255).astype(np.uint8)
    else:  # Softmax output
        mask_processed = np.argmax(mask_pred, axis=-1).astype(np.uint8) * (255 // (mask_pred.shape[-1] - 1))
    
    # Create colored mask
    colored_mask = apply_semantic_colormap(mask_pred)
    mask_pil = Image.fromarray(colored_mask).resize(img.size, Image.Resampling.NEAREST)
    
    # Prepare visualizations based on selected mode
    if viz_mode == "Colored Mask":
        display_img = mask_pil
        caption_text = "Segmentation Mask"
    
    elif viz_mode == "Overlay":
        display_img = create_overlay(img, mask_pil, alpha)
        caption_text = f"Mask Overlay (alpha={alpha})"
    
    elif viz_mode == "Contour Overlay":
        display_img = create_contour_overlay(img, mask_pil)
        caption_text = "Contour Overlay"
    
    else:  # Side by Side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(mask_pil, caption="Segmentation Mask", use_container_width=True)
        st.write(f"*Generated Caption:* {caption}")
        continue  # Skip the single image display below

    # Display single visualization
    if viz_mode != "Side by Side":
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(display_img, caption=caption_text, use_container_width=True)
        st.write(f"*Generated Caption:* {caption}")

    # Add mask statistics
    with st.expander("Mask Statistics"):
        col1, col2, col3 = st.columns(3)
        
        mask_array = np.array(mask_pil)
        unique_colors = np.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
        
        with col1:
            st.metric("Mask Shape", f"{mask_array.shape}")
        with col2:
            st.metric("Unique Colors", f"{len(unique_colors)}")
        with col3:
            coverage = np.mean(mask_array > 0) * 100
            st.metric("Non-zero Coverage", f"{coverage:.1f}%")
        
        # Show color legend
        st.write("Color Legend:")
        legend_colors = {
            "Blue (Low)": "Background or low confidence areas",
            "Green (Medium)": "Medium confidence segmentation",
            "Red (High)": "High confidence segmentation"
        }
        
        for color, meaning in legend_colors.items():
            st.write(f"- {color}: {meaning}")
