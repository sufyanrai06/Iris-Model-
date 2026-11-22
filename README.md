# 🌸 Iris Flower AI Classifier (Pre-trained CLIP Vision Model)

A 100% genuine, pre-trained Vision AI web application powered by **OpenAI CLIP (`openai/clip-vit-base-patch32`)** and **Streamlit**.

## Features
- **Zero Training Needed:** Uses OpenAI's pre-trained CLIP Vision AI directly out of the box.
- **Invalid Image Rejection:** Automatically rejects non-Iris images (cars, faces, roses, random objects) and prompts the user to upload a valid Iris flower.
- **3-Species Identification:** Classifies valid Iris flowers into **Iris Setosa**, **Iris Versicolor**, or **Iris Virginica** with confidence scores.
- **Upload & Live Camera:** Supports both image file uploads and live webcam/mobile camera capture.
- **Cloud Ready:** `requirements.txt` is pre-configured with lightweight CPU PyTorch wheels for free deployment on Streamlit Community Cloud.

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the app:**
   ```bash
   streamlit run app.py
   ```
   *(Note: The first launch downloads the ~600MB CLIP model once and caches it locally).*
