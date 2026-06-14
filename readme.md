# CIFAKE Classifier Web Application

A lightweight, cloud-optimized deployment framework running on Streamlit Cloud that identifies whether an uploaded image file is an authentic camera photograph or an AI-generated synthesis (Stable Diffusion/Midjourney structures).

## How to Deploy to Streamlit Cloud

1. Commit and push all files (`.gitignore`, `requirements.txt`, `data_loader.py`, `model.py`, and `streamlit_app.py`) into a public **GitHub** repository.
2. Sign in to your [Streamlit Cloud Dashboard](https://share.streamlit.io/).
3. Click on **"New App"** and select your GitHub repository.
4. Set the **Main file path** input to `streamlit_app.py`.
5. Click **Deploy!**

*Note: The `kagglehub` library automatically executes the pipeline initialization on the cloud engine's backend without needing manual API credentials for open datasets.*