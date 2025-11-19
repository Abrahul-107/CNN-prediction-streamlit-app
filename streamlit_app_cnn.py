import streamlit as st
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import fitz  # PyMuPDF
import io   # <-- FIXED: required for BytesIO

# ------------------------------------------
# LOAD MODEL (cached)
# ------------------------------------------
@st.cache_resource
def load_classifier_model():
    model = load_model("best_model_fold_4.h5")
    return model

# ------------------------------------------
# PREPROCESS IMAGE
# ------------------------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------------------
# STREAM-PROCESS PDF PAGE-BY-PAGE USING PyMuPDF
# ------------------------------------------
def process_large_pdf_pymupdf(pdf_path, model):
    """
    Efficient, low-RAM page-by-page PDF processing using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    st.success(f"Total pages detected: {total_pages}")

    for page_number in range(total_pages):
        page = doc.load_page(page_number)

        # Convert PDF page to image (pixmap)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")

        # Convert to PIL Image (FIXED HERE)
        pil_image = Image.open(io.BytesIO(img_bytes))

        # Preprocess + predict
        processed = preprocess_image(pil_image)
        pred = model.predict(processed, verbose=0)[0][0]
        label = "FIRST PAGE" if pred >= 0.5 else "NOT FIRST PAGE"

        yield {
            "page_num": page_number + 1,
            "prediction": float(pred),
            "label": label,
            "image": pil_image
        }

        # Free memory
        del pil_image
        del pix

    doc.close()
    tf.keras.backend.clear_session()

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.title("📄 Ultra-Large PDF First-Page Classifier (PyMuPDF Version)")
st.write("Upload a PDF (even **1000+ pages**) and get per-page predictions — without Poppler!")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded_pdf.read())
    temp_pdf.close()

    st.info("Loading model...")
    model = load_classifier_model()

    st.warning("Processing may take time for large PDFs — results will appear live.")

    results_container = st.container()
    progress = st.progress(0)
    status_text = st.empty()

    page_results = []

    for idx, result in enumerate(process_large_pdf_pymupdf(temp_pdf.name, model)):
        page_results.append(result)

        with results_container:
            st.subheader(f"📄 Page {result['page_num']}")
            st.image(result["image"], width=400)

            if result["label"] == "FIRST PAGE":
                st.markdown("### 🟢 FIRST PAGE")
            else:
                st.markdown("### 🔵 NOT FIRST PAGE")

            st.caption(f"Confidence: {result['prediction']:.4f}")
            st.markdown("---")

        progress.progress((idx + 1) / max(1, result["page_num"]))
        status_text.text(f"Processed page {result['page_num']}...")

    st.success("Processing completed!")
