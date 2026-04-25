---
title: Inference Studio
emoji: ⚡
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# ML Lab Assignment | Inference Studio

A premium, highly asynchronous web interface for machine learning model inference. This project serves as a comprehensive playground for testing and interpreting three distinct predictive models using **FastAPI**, **Jinja2**, and **SHAP** explainability.

## Features

- **Multi-Model Support**: Seamlessly switch between Consumer Trends, F1 Pit Strategy, and Sleep Health models.
- **Deep Interpretability**: Real-time **SHAP Diverging Charts** visualizing feature contributions.
- **Neural UI/UX**: Cyber-Industrial aesthetic with glassmorphism, fluid animations, and dark-mode optimization.
- **Data Injection**: "Auto-Inject" feature to quickly test models with real samples from the datasets.
- **HuggingFace Ready**: Fully containerized and optimized for deployment on HF Spaces.

## Project Structure

```text
.
├── dataset/                # Source CSV files from Kaggle
├── inference/
│   ├── app.py              # FastAPI Backend with SHAP Integration
│   └── template/           # Frontend (HTML, CSS, JS)
├── notebooks/              # Jupyter Notebooks for model training
├── outputs/                # Trained model artifacts (.joblib) and performance metrics
├── Dockerfile              # Multi-stage Docker configuration
├── .gitattributes          # LFS tracking for datasets and models
```

## Datasets & Models

| Model | Target | Dataset Source |
|-------|--------|----------------|
| **Consumer Trends** | Category Prediction | [Kaggle Link](https://www.kaggle.com/datasets/minahilfatima12328/consumer-shopping-trends-analysis) |
| **F1 Pit Strategy** | Pit Stop Next Lap | [Kaggle Link](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) |
| **Sleep Health** | Sleep Disorder Classifier | [Kaggle Link](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) |

## Local Development
1. **Setup Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   uvicorn inference.app:app --reload
   ```

3. **Access UI**: Open `http://127.0.0.1:8000`


## Technology Stack
- **Backend**: FastAPI (Python 3.11)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Vanilla CSS, JS (ES6+), Jinja2 Templates
- **Containerization**: Docker
- **ML Engine**: Scikit-Learn, XGBoost, Joblib

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created as part of the B.Tech Semester 04 Machine Learning Lab Assignment.*
