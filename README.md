---
title: Inference Studio
emoji: ⚡
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# Inference Studio

A premium, highly asynchronous web interface for machine learning model inference built with FastAPI and Jinja2.

## Features
- **Triple Model Support**: Switch between Shopping Trends, F1 Strategy, and Sleep Health models.
- **Hot Loading**: Seamless transition between models without page refreshes.
- **Architectural Specs**: Deep dive into dataset distributions and model metrics.
- **Cyber-Industrial UI**: Glassmorphism aesthetic with high-contrast data visualization.
- **Auto-Inject Testing**: Quickly test models with real samples from the datasets.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn inference.app:app --reload
   ```

## Deployment
This project is pre-configured for HuggingFace Spaces.
1. Create a new Space with the Docker SDK.
2. Upload the repository contents.
3. The Space will automatically build and deploy via the included `Dockerfile`.

## Structure
- `inference/app.py`: FastAPI backend with async inference logic.
- `inference/template/`: Frontend assets (HTML, CSS, JS).
- `outputs/`: Trained model artifacts and metrics.
- `dataset/`: Source datasets for feature engineering.
