import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

# Define the filter (example: edge detection mask)
mask = np.array([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]])

st.title('Digit Classification')
st.markdown('---')

st.markdown("""
- Upload Image From Your PC
- Press The Predict Button To Show Result
""")
# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize the image

    # Apply the filter (convolution)
    image_array = cv2.filter2D(image_array, -1, mask)

    # Reshape the image for the model (add batch and channel dimensions)
    image_array = image_array.reshape(1, 28, 28, 1)


    # Show Original and Filtered Image
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(image_array.squeeze(), cmap='gray')
    ax[1].set_title('Filtered Image')
    ax[1].axis('off')
    st.pyplot(fig)


    if st.button("Predict",use_container_width=True):
        prediction = model.predict(image_array).argmax()
        st.subheader(f"Predicted Digit: {prediction}")






# background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.shutterstock.com/image-vector/vector-blue-digital-code-rain-260nw-2482745411.jpg');
        background-size: cover;
        background-position: center;
        height: 100vh;
    }
    </style>
    """, 
    unsafe_allow_html=True
)