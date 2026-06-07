from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")

import matplotlib
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nbclient import NotebookClient
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
NOTEBOOK_PATH = ROOT / "task1_ml_linear_regression.ipynb"
REPORT_PATH = ROOT / "task1_ml_linear_regression_report.pdf"
METRICS_PATH = ROOT / "task1_metrics.json"
COEFFICIENTS_PATH = ROOT / "task1_feature_coefficients.csv"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    dataset = fetch_california_housing(data_home=str(DATA_DIR), as_frame=True)
    frame = dataset.frame.copy()
    frame = frame.rename(columns={"MedHouseVal": "MedianHouseValue"})
    feature_names = list(dataset.feature_names)
    return frame, feature_names


def train_model(frame: pd.DataFrame, feature_names: list[str]):
    x = frame[feature_names]
    y = frame["MedianHouseValue"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("linear_regression", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test - predictions

    metrics = {
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "target_unit": "100,000 USD",
    }

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": model.named_steps[
                "linear_regression"
            ].coef_,
        }
    ).sort_values("standardized_coefficient", key=np.abs, ascending=False)

    return model, x_test, y_test, predictions, residuals, metrics, coefficients


def save_figures(
    frame: pd.DataFrame,
    coefficients: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    residuals: pd.Series,
) -> dict[str, Path]:
    sns.set_theme(style="whitegrid", palette="deep")
    paths = {
        "target_distribution": FIGURE_DIR / "target_distribution.png",
        "correlation_heatmap": FIGURE_DIR / "correlation_heatmap.png",
        "actual_vs_predicted": FIGURE_DIR / "actual_vs_predicted.png",
        "residuals": FIGURE_DIR / "residuals.png",
        "coefficients": FIGURE_DIR / "coefficients.png",
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(frame["MedianHouseValue"], bins=35, kde=True, color="#2c7fb8")
    plt.title("Distribution of Median House Value")
    plt.xlabel("Median house value (100,000 USD)")
    plt.ylabel("District count")
    plt.tight_layout()
    plt.savefig(paths["target_distribution"], dpi=180)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = frame.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(paths["correlation_heatmap"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, s=18, alpha=0.35, edgecolor=None)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], color="#d95f02", linewidth=2)
    plt.title("Actual vs Predicted House Values")
    plt.xlabel("Actual value (100,000 USD)")
    plt.ylabel("Predicted value (100,000 USD)")
    plt.tight_layout()
    plt.savefig(paths["actual_vs_predicted"], dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals, s=18, alpha=0.35, edgecolor=None)
    plt.axhline(0, color="#d95f02", linewidth=2)
    plt.title("Residuals by Predicted Value")
    plt.xlabel("Predicted value (100,000 USD)")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(paths["residuals"], dpi=180)
    plt.close()

    ordered = coefficients.sort_values("standardized_coefficient")
    plt.figure(figsize=(8, 5))
    colors_for_bars = [
        "#1b9e77" if value > 0 else "#7570b3"
        for value in ordered["standardized_coefficient"]
    ]
    plt.barh(ordered["feature"], ordered["standardized_coefficient"], color=colors_for_bars)
    plt.title("Linear Regression Coefficients")
    plt.xlabel("Coefficient after feature standardization")
    plt.tight_layout()
    plt.savefig(paths["coefficients"], dpi=180)
    plt.close()

    return paths


def save_required_artifacts(metrics: dict[str, float], coefficients: pd.DataFrame) -> None:
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)


def notebook_cells() -> list:
    return [
        nbf.v4.new_markdown_cell(
            "# Task 1: Linear Regression House Price Predictor\n\n"
            "This notebook completes the Artificial Intelligence & Machine Learning Task 1 assignment. "
            "It loads the California Housing dataset, performs exploratory data analysis, trains a "
            "linear regression model, evaluates it with MAE, RMSE, and R2, and saves the trained model."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Import libraries\n\n"
            "The workflow uses pandas for tabular analysis, scikit-learn for modeling, and "
            "matplotlib/seaborn for visualizations."
        ),
        nbf.v4.new_code_cell(
            "# Import core libraries for data handling, plotting, modeling, and saving files.\n"
            "from pathlib import Path\n"
            "import json\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.datasets import fetch_california_housing\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "# Use a clean plot style and define reusable project folders.\n"
            "sns.set_theme(style='whitegrid', palette='deep')\n"
            "ROOT = Path.cwd()\n"
            "DATA_DIR = ROOT / 'data'\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Load the California Housing dataset\n\n"
            "The target is `MedHouseVal`, the median house value for a California district. "
            "scikit-learn stores it in units of 100,000 USD."
        ),
        nbf.v4.new_code_cell(
            "# Load the California Housing dataset as a pandas DataFrame.\n"
            "dataset = fetch_california_housing(data_home=str(DATA_DIR), as_frame=True)\n"
            "# Rename the target column so its meaning is easier to read in outputs.\n"
            "df = dataset.frame.copy().rename(columns={'MedHouseVal': 'MedianHouseValue'})\n"
            "feature_names = list(dataset.feature_names)\n\n"
            "# Display the basic dataset size and preview the first rows.\n"
            "print(f'Rows: {df.shape[0]:,}')\n"
            "print(f'Columns: {df.shape[1]}')\n"
            "df.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Explore the data\n\n"
            "This step checks data quality, summary statistics, and the distribution of the target variable."
        ),
        nbf.v4.new_code_cell(
            "# Check whether any columns contain missing values before training.\n"
            "print('Missing values per column:')\n"
            "print(df.isna().sum())\n\n"
            "# Review count, mean, spread, and ranges for every numeric column.\n"
            "df.describe().T"
        ),
        nbf.v4.new_code_cell(
            "# Plot the target distribution to understand typical and capped house values.\n"
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(df['MedianHouseValue'], bins=35, kde=True, color='#2c7fb8')\n"
            "plt.title('Distribution of Median House Value')\n"
            "plt.xlabel('Median house value (100,000 USD)')\n"
            "plt.ylabel('District count')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Correlation helps identify relationships between inputs and the target.\n"
            "plt.figure(figsize=(9, 7))\n"
            "corr = df.corr(numeric_only=True)\n"
            "sns.heatmap(corr, cmap='vlag', center=0, annot=True, fmt='.2f', square=True)\n"
            "plt.title('Feature Correlation Matrix')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Train a Linear Regression model\n\n"
            "The data is split into training and testing sets. A `StandardScaler` is included so coefficients "
            "can be compared on a common scale, followed by `LinearRegression`."
        ),
        nbf.v4.new_code_cell(
            "# Separate input features from the target we want to predict.\n"
            "X = df[feature_names]\n"
            "y = df['MedianHouseValue']\n\n"
            "# Keep 20% of the rows aside for final testing.\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n\n"
            "# StandardScaler normalizes features; LinearRegression learns the prediction rule.\n"
            "model = Pipeline(\n"
            "    steps=[\n"
            "        ('scaler', StandardScaler()),\n"
            "        ('linear_regression', LinearRegression()),\n"
            "    ]\n"
            ")\n"
            "# Fit the model on training data, then predict unseen test data.\n"
            "model.fit(X_train, y_train)\n"
            "predictions = model.predict(X_test)\n\n"
            "print(f'Training rows: {len(X_train):,}')\n"
            "print(f'Testing rows: {len(X_test):,}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Evaluate the model\n\n"
            "MAE and RMSE are in the same target units as the dataset, meaning 1.0 equals about $100,000. "
            "R2 shows the share of test-set variation explained by the model."
        ),
        nbf.v4.new_code_cell(
            "# Calculate standard regression metrics on the test set.\n"
            "mae = mean_absolute_error(y_test, predictions)\n"
            "rmse = np.sqrt(mean_squared_error(y_test, predictions))\n"
            "r2 = r2_score(y_test, predictions)\n\n"
            "# Present metrics in a compact table for the report/notebook.\n"
            "metrics = pd.DataFrame(\n"
            "    {\n"
            "        'Metric': ['MAE', 'RMSE', 'R2'],\n"
            "        'Value': [mae, rmse, r2],\n"
            "    }\n"
            ")\n"
            "metrics"
        ),
        nbf.v4.new_code_cell(
            "# Compare actual values with predictions; points near the diagonal are better.\n"
            "plt.figure(figsize=(7, 6))\n"
            "sns.scatterplot(x=y_test, y=predictions, s=18, alpha=0.35, edgecolor=None)\n"
            "low = min(float(y_test.min()), float(predictions.min()))\n"
            "high = max(float(y_test.max()), float(predictions.max()))\n"
            "# The diagonal line represents perfect predictions.\n"
            "plt.plot([low, high], [low, high], color='#d95f02', linewidth=2)\n"
            "plt.title('Actual vs Predicted House Values')\n"
            "plt.xlabel('Actual value (100,000 USD)')\n"
            "plt.ylabel('Predicted value (100,000 USD)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Residuals show the prediction error for each test example.\n"
            "residuals = y_test - predictions\n\n"
            "plt.figure(figsize=(8, 5))\n"
            "sns.scatterplot(x=predictions, y=residuals, s=18, alpha=0.35, edgecolor=None)\n"
            "# A good residual plot should be roughly centered around zero.\n"
            "plt.axhline(0, color='#d95f02', linewidth=2)\n"
            "plt.title('Residuals by Predicted Value')\n"
            "plt.xlabel('Predicted value (100,000 USD)')\n"
            "plt.ylabel('Residual (actual - predicted)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Interpret coefficients\n\n"
            "Because the model uses standardized features, larger absolute coefficients indicate features "
            "with stronger influence on the linear model."
        ),
        nbf.v4.new_code_cell(
            "# Extract standardized coefficients from the trained linear regression step.\n"
            "coefficients = pd.DataFrame(\n"
            "    {\n"
            "        'feature': feature_names,\n"
            "        'standardized_coefficient': model.named_steps['linear_regression'].coef_,\n"
            "    }\n"
            "# Sort by absolute size so the most influential features appear first.\n"
            ").sort_values('standardized_coefficient', key=np.abs, ascending=False)\n\n"
            "coefficients"
        ),
        nbf.v4.new_code_cell(
            "# Sort coefficients for a horizontal bar chart.\n"
            "ordered = coefficients.sort_values('standardized_coefficient')\n"
            "# Use different colors to distinguish positive and negative coefficients.\n"
            "bar_colors = ['#1b9e77' if value > 0 else '#7570b3' for value in ordered['standardized_coefficient']]\n\n"
            "plt.figure(figsize=(8, 5))\n"
            "plt.barh(ordered['feature'], ordered['standardized_coefficient'], color=bar_colors)\n"
            "plt.title('Linear Regression Coefficients')\n"
            "plt.xlabel('Coefficient after feature standardization')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Improvement ideas\n\n"
            "- Try non-linear models such as Random Forest, Gradient Boosting, or HistGradientBoosting.\n"
            "- Add engineered features, for example rooms per bedroom or location clusters.\n"
            "- Use cross-validation to estimate performance stability.\n"
            "- Investigate capped target values and outliers before production use.\n"
            "- Compare a baseline mean predictor against the trained model."
        ),
    ]


def write_and_execute_notebook() -> None:
    runtime_dir = ROOT / ".jupyter_runtime"
    runtime_dir.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    notebook["cells"] = notebook_cells()

    nbf.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=240,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(notebook, NOTEBOOK_PATH)


def add_wrapped_text(
    fig,
    x: float,
    y: float,
    text: str,
    *,
    width: int = 95,
    size: int = 10,
    weight: str = "normal",
    color: str = "#1f2933",
    line_gap: float = 0.026,
) -> float:
    for line in textwrap.wrap(text, width=width):
        fig.text(x, y, line, fontsize=size, fontweight=weight, color=color, va="top")
        y -= line_gap
    return y


def add_page_footer(fig, page_number: int) -> None:
    fig.text(0.92, 0.035, f"Page {page_number}", ha="right", fontsize=8, color="#5f6c7b")


def add_image(fig, image_path: Path, rect: list[float]) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(image_path))
    ax.axis("off")


def add_table(fig, data: list[list[str]], rect: list[float]) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    table = ax.table(cellText=data[1:], colLabels=data[0], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c9d3df")
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6f9")


def new_report_page(title: str, page_number: int):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(
        0.08,
        0.94,
        title,
        fontsize=18,
        fontweight="bold",
        color="#17324d",
        va="top",
    )
    add_page_footer(fig, page_number)
    return fig


def build_report(
    frame: pd.DataFrame,
    feature_names: list[str],
    metrics: dict[str, float],
    coefficients: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    with PdfPages(REPORT_PATH) as pdf:
        fig = new_report_page("Task 1: Linear Regression House Price Predictor", 1)
        y = 0.875
        y = add_wrapped_text(
            fig,
            0.08,
            y,
            "Objective",
            width=80,
            size=13,
            weight="bold",
            color="#17324d",
        )
        y = add_wrapped_text(
            fig,
            0.08,
            y - 0.008,
            "This report summarizes a complete machine learning workflow using the "
            "California Housing dataset. The goal is to predict median district house "
            "value from demographic, geographic, and household features using a Linear "
            "Regression model.",
        )
        y = add_wrapped_text(
            fig,
            0.08,
            y - 0.02,
            "Dataset Snapshot",
            width=80,
            size=13,
            weight="bold",
            color="#17324d",
        )
        y = add_wrapped_text(
            fig,
            0.08,
            y - 0.008,
            f"The dataset contains {len(frame):,} rows, {len(feature_names)} input "
            "features, and one target column. The target is median house value, stored "
            "in units of 100,000 USD. The scikit-learn dataset has no missing values.",
        )
        y = add_wrapped_text(
            fig,
            0.08,
            y - 0.02,
            "Modeling Approach",
            width=80,
            size=13,
            weight="bold",
            color="#17324d",
        )
        add_wrapped_text(
            fig,
            0.08,
            y - 0.008,
            "The data was split into 80 percent training and 20 percent testing sets "
            "with a fixed random seed for reproducibility. The pipeline standardized "
            "features with StandardScaler, then trained a LinearRegression model.",
        )
        metric_rows = [
            ["Metric", "Value", "Interpretation"],
            ["MAE", f"{metrics['mae']:.3f}", "Average absolute error in 100,000 USD units"],
            ["RMSE", f"{metrics['rmse']:.3f}", "Penalizes larger prediction errors"],
            ["R2", f"{metrics['r2']:.3f}", "Share of test-set variance explained"],
        ]
        fig.text(0.08, 0.43, "Test Set Metrics", fontsize=13, fontweight="bold", color="#17324d")
        add_table(fig, metric_rows, [0.08, 0.23, 0.84, 0.18])
        add_wrapped_text(
            fig,
            0.08,
            0.185,
            f"The model achieved an R2 of {metrics['r2']:.3f}. This is a useful baseline, "
            "but the remaining error indicates that linear relationships alone do not "
            "capture all housing price patterns.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_report_page("Exploratory Data Analysis", 2)
        add_wrapped_text(
            fig,
            0.08,
            0.875,
            "Median income is the strongest positive signal for house value. Location "
            "features also matter, and several predictors show correlation, which is "
            "expected for geographic housing data.",
        )
        add_image(fig, figure_paths["target_distribution"], [0.08, 0.50, 0.84, 0.30])
        add_image(fig, figure_paths["correlation_heatmap"], [0.10, 0.08, 0.80, 0.42])
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_report_page("Model Diagnostics", 3)
        add_wrapped_text(
            fig,
            0.08,
            0.875,
            "Predicted values generally follow actual values, but the diagnostic plots "
            "show spread around the ideal line. This suggests room for non-linear "
            "models, feature engineering, and closer handling of high-value capped "
            "observations.",
        )
        add_image(fig, figure_paths["actual_vs_predicted"], [0.14, 0.48, 0.72, 0.35])
        add_image(fig, figure_paths["residuals"], [0.08, 0.12, 0.84, 0.30])
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_report_page("Feature Influence and Improvement Ideas", 4)
        add_wrapped_text(
            fig,
            0.08,
            0.875,
            "The five largest standardized coefficients by absolute value are shown "
            "below. Coefficients are directional signals for this linear model, not "
            "proof of causation.",
        )
        coefficient_rows = [["Feature", "Standardized Coefficient"]] + [
            [row.feature, f"{row.standardized_coefficient:.3f}"]
            for row in coefficients.head(5).itertuples(index=False)
        ]
        add_table(fig, coefficient_rows, [0.18, 0.63, 0.64, 0.18])
        add_image(fig, figure_paths["coefficients"], [0.08, 0.29, 0.84, 0.29])
        fig.text(0.08, 0.235, "Improvement Ideas", fontsize=13, fontweight="bold", color="#17324d")
        ideas = [
            "Test tree-based models such as Random Forest or Gradient Boosting.",
            "Add engineered features such as rooms per bedroom or location clusters.",
            "Use cross-validation to estimate performance stability.",
            "Investigate capped target values and outliers before production use.",
            "Compare against a simple baseline mean predictor.",
        ]
        y = 0.205
        for idea in ideas:
            y = add_wrapped_text(fig, 0.10, y, f"- {idea}", width=85, line_gap=0.024)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    frame, feature_names = load_dataset()
    model, x_test, y_test, predictions, residuals, metrics, coefficients = train_model(
        frame, feature_names
    )
    figure_paths = save_figures(frame, coefficients, y_test, predictions, residuals)
    save_required_artifacts(metrics, coefficients)
    write_and_execute_notebook()
    build_report(frame, feature_names, metrics, coefficients, figure_paths)

    print("Generated deliverables:")
    for path in [
        NOTEBOOK_PATH,
        REPORT_PATH,
        METRICS_PATH,
        COEFFICIENTS_PATH,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
