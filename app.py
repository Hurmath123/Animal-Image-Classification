import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2
import os

# Set page config
st.set_page_config(page_title="Animal Classifier", layout="centered")

# Custom styling
st.markdown("""
<style>
h1 {
    color: #4CAF50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Animal Image Classifier")
st.markdown("Upload an image to identify the animal using a fine-tuned MobileNetV2 model.")

# Load model
@st.cache_resource
def load_local_model():
    model = load_model("mobilenetv2_best.h5")
    return model

model = load_local_model()

# Class labels
CLASS_NAMES = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Tiger', 'Zebra'
]

# Grad-CAM utility
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# Upload image
uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)[0]
        top3_idx = np.argsort(pred)[::-1][:3]

        st.subheader("Top Predictions")
        for i in top3_idx:
            st.write(f"**{CLASS_NAMES[i]}:** {pred[i] * 100:.2f}%")

        # Confidence bar chart
        st.subheader("Confidence Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(CLASS_NAMES, pred, color="skyblue")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

        # Optional Grad-CAM
        if st.checkbox("Show Grad-CAM Heatmap"):
            heatmap = make_gradcam_heatmap(x, model)
            gradcam = display_gradcam(img, heatmap)
            st.image(gradcam, caption="Grad-CAM Heatmap", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")
