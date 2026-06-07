# California Housing Price Prediction - Linear Regression

This project completes **Artificial Intelligence & Machine Learning - Task 1**. It builds and evaluates a Linear Regression model to predict California district median house values using the California Housing dataset from scikit-learn.

## Objective

The goal is to demonstrate a complete beginner-friendly machine learning workflow:

- Load the California Housing dataset
- Explore the data with summary statistics and visualizations
- Preprocess the features
- Train a Linear Regression model
- Evaluate the model using MAE, RMSE, and R2 score
- Create a short PDF report

## Dataset

The project uses the built-in scikit-learn California Housing dataset.

Target column:

- `MedianHouseValue`: median district house value in units of `$100,000`

Input features:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

## Model

The model is a scikit-learn pipeline:

1. `StandardScaler` for feature scaling
2. `LinearRegression` for prediction

The dataset is split into:

- 80% training data
- 20% testing data

## Results

Test set performance:

| Metric | Value |
| --- | ---: |
| MAE | 0.533 |
| RMSE | 0.746 |
| R2 Score | 0.576 |

Because the target is measured in units of `$100,000`, an MAE of `0.533` means the average absolute error is about `$53,300`.

## Project Files

| File / Folder | Purpose |
| --- | --- |
| `task1_ml_linear_regression.ipynb` | Main Jupyter Notebook with code, comments, plots, model training, and evaluation |
| `task1_ml_linear_regression_report.pdf` | Short 4-page PDF report summarizing EDA, model, metrics, and improvement ideas |
| `figures/` | Generated EDA and evaluation plots |
| `task1_metrics.json` | Saved model metrics |
| `task1_feature_coefficients.csv` | Saved linear regression coefficients |
| `generate_task1_deliverables.py` | Reproducible script that regenerates notebook, report, figures, and metrics |
| `requirements.txt` | Python packages required to run the project |

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Regenerate all deliverables:

```bash
python generate_task1_deliverables.py
```

## Improvement Ideas

- Try non-linear models such as Random Forest or Gradient Boosting
- Add engineered features such as rooms per bedroom or location clusters
- Use cross-validation for more stable performance estimates
- Investigate capped target values and outliers
- Compare results against a simple baseline model

## GitHub Upload Checklist

Recommended files to upload:

- `README.md`
- `requirements.txt`
- `task1_ml_linear_regression.ipynb`
- `task1_ml_linear_regression_report.pdf`
- `generate_task1_deliverables.py`
- `task1_metrics.json`
- `task1_feature_coefficients.csv`
- `figures/`

Do not upload:

- `data/`
- `models/`
- `.ipynb_checkpoints/`
- `.jupyter_runtime/`
- Any temporary cache files
