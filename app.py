import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import uuid
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Digit Classifier", page_icon="🧠", layout="centered")

# --- Model Loading ---
@st.cache_resource
def get_model(model_path="model1.keras"):
    """
    Loads the Keras model.
    Checks if the model file exists before attempting to load it.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory as your app.")
        st.stop()  # Stop app execution if model file is missing
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        st.stop()  # Stop app if model loading fails

model = get_model()

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("Instructions")
    st.write("""
    - Draw a digit (0-9) using your mouse or touchscreen on the black canvas.
    - Select 'Draw' to add lines or 'Erase' to remove them.
    - Click the 'Predict' button to classify the drawn digit using AI.
    - Click 'Clear Canvas' to reset the drawing area.
    """)

# --- Main Title and Subtitle ---
st.markdown("<h1 style='text-align:center; color:#00f5d4;'>🖌️ AI Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa;'>Draw your digit below and see what the AI predicts!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Controls (Mode, Clear, Predict) ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    mode = st.radio("Mode", ["Draw", "Erase"], horizontal=True, help="Choose between drawing or erasing on the canvas.")

with col2:
    clear = st.button("Clear Canvas", help="Click to clear all drawings from the canvas.")

with col3:
    predict_btn = st.button("Predict", help="Click to ask the AI to predict the digit you've drawn.")

# --- Canvas Key Management ---
if clear:
    st.session_state["canvas_key"] = str(uuid.uuid4())
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = str(uuid.uuid4())

# --- Canvas Setup ---
stroke_col = "#FFFFFF" if mode == "Draw" else "#000000"

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color=stroke_col,
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state["canvas_key"],
    display_toolbar=True,
)

# --- Prediction Logic ---
if predict_btn:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        if np.sum(img) < 1000:
            st.warning("The canvas appears empty. Please draw a digit before predicting.")
        else:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized / 255.0
                input_img = normalized.reshape(1, 28, 28, 1)

                pred = model.predict(input_img)
                digit = int(np.argmax(pred))
                confidence = float(np.max(pred)) * 100

                st.success(f"Predicted Digit: **{digit}**")
                st.info(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)
    else:
        st.warning("Please draw something on the canvas before predicting.")
