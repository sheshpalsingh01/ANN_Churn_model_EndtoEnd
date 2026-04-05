# ANN Customer Churn Prediction

This project implements an Artificial Neural Network (ANN) to predict customer churn based on a dataset of bank customers. It includes a Streamlit web application for interactive predictions and TensorBoard integration for visualizing training metrics.

## Project Structure
- `app.py`: Streamlit application for inference and visualization.
- `01_modelprep.ipynb`: Jupyter notebook for data preprocessing and model training.
- `04_model.h5`: Trained Keras ANN model.
- `01_encoded_gender.pkl`, `02_encoded_geo.pkl`, `03_scaler.pkl`: Preprocessing artifacts for gender encoding, geography encoding, and feature scaling.
- `Churn_Modelling.csv`: The dataset used for training and testing.
- `requrement.txt`: List of dependencies.
- `logs/`: Directory containing TensorBoard training logs.

## Setup & Installation

1. **Clone the repository** (or navigate to this directory).
2. **Install dependencies**:
   ```bash
   pip install -r requrement.txt
   ```

## Running the Application

To start the Streamlit application, run:
```bash
streamlit run app.py
```

## Features
- **Real-time Prediction**: Input customer details like Credit Score, Geography, Gender, Age, Balance, etc., to get the probability of churn.
- **Visual Feedback**: Real-time probability display with success/warning alerts.
- **TensorBoard Integration**: View training performance metrics directly within the Streamlit dashboard.

## Documentation
- [Streamlit UI](Streamlit.pdf)


