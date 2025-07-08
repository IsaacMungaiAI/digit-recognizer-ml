import streamlit as st
from scipy.ndimage import gaussian_filter
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import joblib

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠ML driven Digit Recognizer")

# Load model trained on sklearn.datasets.load_digits()
model = joblib.load("model.pkl")

st.write("Draw a digit (0–9) below:")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",          # white digit
    background_color="black",      # black background
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict when button is clicked
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Extract grayscale image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))

        # Resize to 8x8 (like digits dataset)
        img = img.resize((8, 8), Image.Resampling.LANCZOS)

        # Convert to array and scale 0–255 → 0–16
        img_array = np.array(img).astype(np.float64)
        img_array = (img_array / 255.0) * 16



        img_array = gaussian_filter(img_array, 0.5)
        img_array[img_array < 1.0] = 0.0

        # Visual debug
        st.image(img_array, caption="Preprocessed Input", width=150, clamp=True)
        st.text("Pixel values:")
        st.text(np.round(img_array, 1))

        # Flatten and predict
        img_flat = img_array.flatten().reshape(1, -1)
        prediction = model.predict(img_flat)[0]
        st.success(f"Predicted Digit: {prediction}")
    else:
        st.warning("Please draw a digit first.")
