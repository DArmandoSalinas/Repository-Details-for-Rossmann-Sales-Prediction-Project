# Rossmann Sales Prediction with Feedforward Neural Network

## Overview
This project predicts daily sales for Rossmann stores using a Feedforward Neural Network (FNN). By leveraging structured data and advanced deep learning techniques, the model identifies complex patterns to improve sales forecasts, aiding store managers in better scheduling and decision-making.

## Features
### Data Preprocessing
- **Handling Missing Data**:
  - Fills missing values for competition distance, promo intervals, and other fields with appropriate defaults.
- **Feature Engineering**:
  - Extracted temporal attributes (Year, Month, WeekOfYear).
  - Calculated competition and promotion duration metrics.
- **Integration**:
  - Merged `store.csv` with training and testing datasets to include store-specific details.
- **Column Selection**:
  - Removed irrelevant or redundant features like `Date` and `Customers`.

### Feedforward Neural Network (FNN)
- **Architecture**:
  - Input Layer: Preprocessed features (e.g., Year, Month, Promo2Open).
  - Hidden Layers:
    - Three dense layers with nodes (128, 64, 32).
    - ReLU activation and dropout regularization.
  - Output Layer: A single neuron for continuous sales predictions.
- **Optimization**:
  - Loss Function: Mean Squared Error (MSE).
  - Optimizer: Adam for adaptive learning rates.
  - Early stopping to prevent overfitting.

## How It Works
1. **Data Preprocessing**:
   - Uses the `Preprocessed.py` file to clean and transform the data for training.
2. **Training**:
   - Trains the FNN on historical sales data with 100 epochs and a batch size of 32.
3. **Evaluation**:
   - Validates predictions using the Kaggle test dataset.
4. **Prediction**:
   - Saves predictions to a file for submission on Kaggle.

## Results
- The FNN model effectively captures non-linear relationships in sales data.
- Predictions are saved in a CSV file and uploaded to Kaggle for evaluation.

## Demonstration
Key steps for running the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/DArmandoSalinas/RossmannSales_FNN.git
2. cd RossmannSales_FNN
3. python Preprocessed.py
4. python ExploringTrainingTesting.py
