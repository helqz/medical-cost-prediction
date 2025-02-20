# Medical Cost Prediction with Keras

Predict medical insurance costs using a neural network. This project demonstrates an end-to-end machine learning workflow for regression tasks.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)

## Project Overview
- **Goal**: Predict medical charges based on health metrics (age, BMI, smoking status, etc.).
- **Dataset**: 1,338 samples with 7 features ([source](https://www.kaggle.com/mirichoi0218/insurance)).
- **Workflow**: EDA → Data preprocessing → Neural network training → Model evaluation.

## Key Features
- Early stopping to prevent overfitting
- Comparison of actual vs predicted costs

## Installation
```bash
git clone https://github.com/helqz/medical-cost-prediction.git
cd medical-cost-prediction
pip install -r requirements.txt  # Install dependencies
```

## Usage
1. Run Jupyter Notebook:
    ```bash
    jupyter notebook medical_cost_prediction.ipynb
    ```
2. Key steps:
    - Data preprocessing and visualization
    - Model training with Keras
    - Performance evaluation (MAE: ~$2,700)
      
## Model Architecture
```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
```

## Key Findings
- 🚬 Smokers incur 3-4x higher costs than non-smokers
- 📈 Age and BMI show moderate positive correlations
- 🌍 Region/sex have minimal impact

## Limitations
- Small dataset (~1,300 samples)
- Lacks detailed medical history/pre-existing conditions
