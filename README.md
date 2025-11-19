
# PDF First-Page Classifier: Streamlit App

A fast and efficient Streamlit web application that automatically detects whether any page in a PDF is the **first page** (cover page) of a document using a trained deep learning model.

Built for speed and reliability, the app handles **very large PDFs** without running out of memory by processing pages one at a time.

## Features

- Upload any PDF (no size limit in practice)
- Fast page rendering with **PyMuPDF (fitz)**
- On-the-fly preprocessing and inference using a **TensorFlow/Keras CNN**
- Page-by-page classification: **First Page** vs **Not First Page**
- Confidence scores displayed for each page
- Live page preview in the browser
- No Poppler dependency → works perfectly on Windows, macOS, and Linux
- Memory-efficient streaming processing

## How It Works

1. The uploaded PDF is opened with **PyMuPDF**
2. Each page is rendered to an image (300 DPI by default)
3. The image is preprocessed to match the training conditions
4. The loaded Keras model runs inference
5. Results (label + confidence) are streamed back to the browser immediately
6. Memory is released after each page → handles 1000+ page PDFs smoothly

## Tech Stack

- Python 3.10+
- Streamlit
- PyMuPDF (`fitz`)
- TensorFlow / Keras
- Pillow
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
streamlit run streamlit_app_v2.py
```

The app will open in your browser (usually at http://localhost:8501).

## Quick Start with Docker (Recommended)


### 1. Clone the repository (or copy the files into a folder)
```bash
 git clone https://github.com/Abrahul-107/CNN-prediction-streamlit-app.git
 ```
 ```shell
cd CNN-prediction-streamlit-app
```

### 2. Build the Docker image
```bash
docker build -t document-intelligence-app .
```

### 3. Run the container
```docker run -p 8501:8501 document-intelligence-app```

Open your browser and go to: http://localhost:8501

Boom You're ready!

## Why PyMuPDF instead of pdf2image?

| Feature                   | PyMuPDF (fitz)       | pdf2image + Poppler          |
|---------------------------|----------------------|------------------------------|
| Windows support           | Native               | Requires Poppler install     |
| Installation ease         | `pip install pymupdf`| Complex on Windows           |
| Speed                     | Very fast            | Slower                       |
| Memory usage              | Low                  | High (loads full PDF at once)   |
| External dependencies     | None                 | Poppler binary               |

## Use Cases

- Automated detection of title/cover pages
- Preprocessing step for RAG and document AI pipelines
- Organizing large collections of scanned documents
- Batch processing in document management systems
- Training data preparation for layout analysis models

## Model

The repository expects a Keras `.h5` model trained on binary classification (first page vs subsequent pages).  
Example training notebook available in `notebooks/` (if you include one).

## Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or feature requests
- Submit pull requests
- Improve documentation or add tests

Please follow standard GitHub flow and write clear commit messages.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact / Support

For questions, integration help, or custom extensions (e.g., multi-class layout detection), open an issue .
