import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="OCR Text Detection",
    page_icon="📄",
    layout="wide"
)

st.title("OCR Text Detection System")
st.write("Upload an image to detect and extract text.")

@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_array)

    if st.button("Extract Text"):

        with st.spinner("Detecting text..."):
            results = reader.readtext(image_array)

            output = image_array.copy()
            extracted_text = []

            for bbox, text, confidence in results:

                extracted_text.append(text)

                points = np.array(bbox, dtype=np.int32)

                cv2.polylines(
                    output,
                    [points],
                    True,
                    (0, 255, 0),
                    2
                )

            final_text = "\n".join(extracted_text)

        with col2:
            st.subheader("Detected Text")
            st.image(output)

        st.subheader("Extracted Text")
        st.text_area("Result", final_text, height=200)

        st.download_button(
            "Download Text",
            final_text,
            file_name="ocr_result.txt"
        )