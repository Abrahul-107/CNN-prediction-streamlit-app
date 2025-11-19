import streamlit as st
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import fitz   # PyMuPDF
import io

# -------------------------
# LOAD MODEL (CPU ONLY)
# -------------------------
@st.cache_resource
def load_classifier_model():
    return load_model("best_model_fold_4.h5")   # <-- replace with your model


# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------
# CLASSIFY PDF IN STREAMING MODE (PyMuPDF)
# -------------------------
def classify_large_pdf(pdf_path, model, batch_size=20):
    """
    Processes PDF in chunks to avoid RAM explosion for 1000+ pages.
    Now uses PyMuPDF instead of pdf2image.
    """

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    st.write("📘 Total Pages Detected:", total_pages)

    results = []
    progress = st.progress(0)

    page_start = 0

    # Process in batches
    while page_start < total_pages:

        page_end = min(page_start + batch_size, total_pages)

        for page_num in range(page_start, page_end):

            page = doc.load_page(page_num)

            pix = page.get_pixmap(dpi=120)
            img_bytes = pix.tobytes("png")

            pil_image = Image.open(io.BytesIO(img_bytes))

            # Predict
            processed = preprocess_image(pil_image)
            pred = model.predict(processed, verbose=0)[0][0]

            label = "FIRST PAGE" if pred >= 0.5 else "NOT FIRST PAGE"

            results.append({
                "page_num": page_num + 1,
                "prediction": float(pred),
                "label": label,
            })

            # Progress bar
            progress.progress((page_num + 1) / total_pages)

            del pil_image
            del pix

        page_start += batch_size

    progress.progress(1.0)
    doc.close()
    return results


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📄PDF First-Page Classifier")
st.write("Christian's Streamlit app for classifying first pages in large PDFs.")
# st.write("Upload a PDF (up to **1000+ pages**) — no Poppler needed!")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf is not None:
    st.info("Extracting & classifying pages... Please wait.")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(uploaded_pdf.read())
    temp_pdf.close()

    model = load_classifier_model()

    results = classify_large_pdf(temp_pdf.name, model, batch_size=30)

    st.success("Completed!")

    # -------------------------
    # SUMMARY TABLE
    # -------------------------
    st.subheader("📊 Summary Results")

    first_pages = [r for r in results if r["label"] == "FIRST PAGE"]
    st.write(f"**Detected FIRST PAGE count: {len(first_pages)}**")

    # Show first few candidates
    st.table(first_pages[:10])

    # -------------------------
    # Expandable full results
    # -------------------------
    with st.expander("🔍 View Full Page-by-Page Results"):
        st.dataframe(results)
