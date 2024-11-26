import numpy as np
import cv2
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

# Define the filter (example: edge detection mask)
mask = np.array([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]])

st.markdown("<h1 style='color: white;'>Digit Classification</h1>", unsafe_allow_html=True)
st.markdown('---')

st.markdown(
    """
    <p style="color: white;">
    - Upload Image From Your PC<br>
    - Press The Predict Button To Show Result
    </p>
    """, 
    unsafe_allow_html=True
)

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    image_array = image.convert("L")
    image_array = image_array.resize((28, 28))  
    image_array = np.array(image_array) / 255.0  


    # Apply the filter (convolution)
    image_array = cv2.filter2D(image_array, -1, mask)

    # Reshape the image for the model (add batch and channel dimensions)
    image_array = image_array.reshape(1, 28, 28, 1)


    # Show Original and Filtered Image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(image_array.squeeze(), cmap='gray')
    ax[1].set_title('Filtered Image')
    ax[1].axis('off')
    st.pyplot(fig)

    if st.button("Predict",use_container_width=True):
        prediction = model.predict(image_array).argmax()
        st.markdown(f"<h2 style='color: #4a7c59;'>Predicted Digit: {prediction}</h2>", unsafe_allow_html=True)






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
