# 🏡 House Price AI — California House Price Prediction

A machine-learning project that predicts **California median house values** from location, housing and income features, with an interactive **Streamlit web app** for making predictions, comparing models and exploring the data.

![Dashboard](screenshots/dashboard.png)

---

## 📌 Table of Contents

1. [Project Description](#-project-description)
2. [Why I Chose This Project](#-why-i-chose-this-project)
3. [Dataset](#-dataset)
4. [Project Workflow](#-project-workflow)
5. [Models Used](#-models-used)
6. [Evaluation Metrics & Results](#-evaluation-metrics--results)
7. [Streamlit App Features](#-streamlit-app-features)
8. [Technologies Used](#-technologies-used)
9. [Project Structure](#-project-structure)
10. [Installation & Usage](#-installation--usage)
11. [Limitations](#-limitations)
12. [Future Improvements](#-future-improvements)
13. [Author](#-author)

---

## 📖 Project Description

Housing prices depend on many things at once: how close a home is to the ocean, the income of the neighbourhood, how crowded it is, how old the buildings are, and more. This project learns those relationships from real data and turns them into a price estimate.

The project covers the full machine-learning lifecycle:

- Exploratory data analysis (EDA) of 20,640 California block groups
- Data cleaning, feature engineering and feature selection
- Scikit-learn **pipelines** that bundle preprocessing and the model together
- Hyperparameter tuning with 5-fold cross-validation (`GridSearchCV`)
- Evaluation with MAE, MSE, RMSE and R², plus overfitting checks
- Deployment as an interactive **Streamlit** application

Two regression models are trained and compared: a **Decision Tree Regressor** and **Multiple Linear Regression**.

---

## 💡 Why I Chose This Project

> ✏️ *Edit this section so it is in your own words — it is the part reviewers and recruiters read most closely.*

- **Real-world relevance:** price estimation is a classic, practical regression problem used in real estate, banking and insurance.
- **Complete end-to-end practice:** it let me work through every stage of a data-science project, from raw CSV to a deployed app, instead of stopping at a notebook.
- **Beginner-friendly but meaningful dataset:** the California Housing dataset is well known, has a mix of numeric and categorical features, and contains missing values and outliers, which are realistic data-cleaning challenges.
- **Model comparison:** comparing a simple, interpretable model (Linear Regression) with a non-linear one (Decision Tree) showed me the trade-offs between accuracy, interpretability and overfitting.
- **Skill building:** it strengthened my skills in Python, Pandas, Scikit-learn pipelines, model evaluation and building user-facing apps with Streamlit.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Name | California Housing Prices (`housing.csv`) |
| Source | 1990 US Census data for California block groups *(add your download link, e.g. Kaggle)* |
| Records | **20,640** rows |
| Columns | **10** (9 input features + 1 target) |
| Target | `median_house_value` (US dollars) |
| Missing values | `total_bedrooms` — 207 rows (≈1%) |
| Duplicate rows | 0 |

**Features**

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic location of the block group |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms in the block |
| `total_bedrooms` | Total number of bedrooms in the block |
| `population` | Number of people living in the block |
| `households` | Number of households in the block |
| `median_income` | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Category: `<1H OCEAN` (9,136), `INLAND` (6,551), `NEAR OCEAN` (2,658), `NEAR BAY` (2,290), `ISLAND` (5) |

**Target statistics:** mean ≈ $206,856 · median ≈ $179,700 · min $14,999 · max $500,001.
Prices are **capped at $500,001** in the original data (965 records, ≈4.7%), which affects how well any model can predict very expensive areas.

---

## 🔄 Project Workflow

1. **Data loading & inspection** – shape, data types, memory usage, summary statistics.
2. **Data quality checks** – missing values, duplicates, unique values, skewness.
3. **Exploratory data analysis** – histograms, box plots, correlation heatmap, scatter plots (income vs price, rooms vs price, etc.) and price by ocean proximity.
4. **Train/test split** – 80% training (16,512 rows) and 20% testing (4,128 rows), `random_state=42`.
5. **Outlier inspection** – IQR analysis on the training set (outliers were inspected but not removed, so the models see realistic data).
6. **Feature engineering** – three new ratio features:
   - `rooms_per_household = total_rooms / households`
   - `bedrooms_per_room = total_bedrooms / total_rooms`
   - `population_per_household = population / households`
7. **Preprocessing pipelines**
   - Numeric: median imputation (+ `StandardScaler` for Linear Regression only)
   - Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
8. **Feature selection** – `SelectKBest(f_regression)`; `median_income` was the strongest predictor by far (F-score ≈ 15,057).
9. **Hyperparameter tuning** – `GridSearchCV` with 5-fold cross-validation, scored on RMSE.
10. **Evaluation** – MAE, MSE, RMSE, R², actual-vs-predicted plots, residual analysis, train-vs-test comparison.
11. **Model export** – full pipelines saved with `joblib`.
12. **Deployment** – Streamlit web application.

---

## 🤖 Models Used

### 🌳 Decision Tree Regressor
Learns a series of if/else rules from the features, so it can capture **non-linear** relationships and interactions.

- Search space: `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 10}, `min_samples_leaf` ∈ {1, 5}, `k` ∈ {8, 10, all} — 36 candidates × 5 folds = 180 fits
- **Best parameters:** `max_depth=10`, `min_samples_leaf=5`, `min_samples_split=2`, all 16 features kept

### 📈 Multiple Linear Regression
Learns a weighted sum of the features. Simple, fast and easy to interpret, but it can only model **linear** relationships.

- Search space: `fit_intercept` ∈ {True, False}, `k` ∈ {8, 10, all} — 6 candidates × 5 folds = 30 fits
- **Best parameters:** `fit_intercept=True`, all 16 features kept

Both models are stored as **complete scikit-learn pipelines** (preprocessing → feature selection → model), so raw feature values can be passed straight to `predict()`.

---

## 📊 Evaluation Metrics & Results

Metrics are computed on the **held-out test set (4,128 samples)** that the models never saw during training or tuning.

| Model | MAE (USD) | MSE | RMSE (USD) | R² Score |
|---|---:|---:|---:|---:|
| **Decision Tree Regressor** ⭐ | **39,871.31** | **3,630,558,063.58** | **60,254.11** | **0.7229** |
| Multiple Linear Regression | 49,645.49 | 4,778,547,424.03 | 69,127.04 | 0.6353 |

**Best model: Decision Tree Regressor** — its RMSE is about **12.8% lower** and its MAE about **19.7% lower** than Linear Regression, and it explains about **72.3%** of the variation in house prices (vs. 63.5%).

> **A note on "accuracy":** this is a *regression* problem, so classification accuracy does not apply. The equivalent measure is **R²** (share of price variation explained). R² = 0.72 means the Decision Tree explains roughly 72% of the variation in prices. On average its predictions are off by about **$39.9k** (MAE), which is roughly 19% of the mean house price.

### What the metrics mean

| Metric | Meaning | Better is |
|---|---|---|
| **MAE** | Average absolute error in dollars | Lower |
| **MSE** | Average squared error (in dollars²) — punishes large errors | Lower |
| **RMSE** | Square root of MSE, in dollars | Lower |
| **R² Score** | Share of price variation explained by the model | Higher (max 1.0) |

### Cross-validation vs. test performance

| Model | CV RMSE (5-fold) | Test RMSE |
|---|---:|---:|
| Decision Tree | 60,265.17 | 60,254.11 |
| Multiple Linear Regression | 67,858.78 | 69,127.04 |

The cross-validation and test errors are very close for both models, which suggests the results are stable and the test score is a realistic estimate rather than a lucky split.

### Overfitting check

| Model | Training R² | Testing R² | Gap |
|---|---:|---:|---:|
| Decision Tree | 0.8307 | 0.7229 | 0.1078 |
| Multiple Linear Regression | 0.6582 | 0.6353 | 0.0229 |

The Decision Tree shows a moderate train/test gap (some overfitting, kept in check by `max_depth=10` and `min_samples_leaf=5`) but still generalises better overall. Linear Regression is very consistent but **underfits**: the relationship between features and price is not purely linear.

### What drives the price?

- **Decision Tree (feature importance):** `median_income` (≈ 0.55) is by far the most important, followed by `ocean_proximity_INLAND` (≈ 0.17), `population_per_household` (≈ 0.12), then `longitude` and `latitude`.
- **Linear Regression (coefficients on scaled features):** `median_income` raises the predicted price the most among numeric features, while `INLAND` location, `latitude` and `longitude` lower it. The large `ISLAND` coefficient should be treated with caution because only 5 records have that category.

---

## 🖥️ Streamlit App Features

- **Predict a price** from location, property details, income and ocean proximity — with **Decision Tree**, **Linear Regression**, or **Compare Both** side by side
- **Sample profiles** (coastal, Bay Area, city, inland) and one-click reset
- **Result dashboard:** estimate, gauge, price percentile, comparison with the average for the chosen ocean-proximity group, and a typical-error range based on test MAE
- **What-if analysis:** see how the price changes as one feature varies
- **Location view:** map of the property plus the 10 nearest homes in the dataset
- **Model Comparison:** metrics table, charts, actual-vs-predicted, residuals and feature importance
- **Data Explorer:** filters, price map, distributions, correlation heatmap, CSV download
- **Prediction History:** session log with chart and CSV download
- Custom green / tangerine-yellow / ivory theme

---

## 🛠️ Technologies Used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn (notebook), Plotly (app) |
| Machine learning | Scikit-learn (Pipelines, ColumnTransformer, GridSearchCV, SelectKBest) |
| Model persistence | Joblib |
| Web app | Streamlit |
| Environment | Google Colab (training), VS Code (app development), Git & GitHub |

---

## 📁 Project Structure

```
House_Price_Prediction/
├── .streamlit/
│   └── config.toml                  # app colour theme
├── data/
│   └── housing.csv                  # dataset
├── models/
│   ├── decision_tree_pipeline.pkl   # trained Decision Tree pipeline
│   └── linear_regression_pipeline.pkl  # trained Linear Regression pipeline
├── notebooks/
│   └── House_Price_Prediction.ipynb # EDA, training, tuning, evaluation
├── screenshots/
│   └── dashboard.png
├── app.py                           # Streamlit application
├── model_metrics.csv                # test-set metrics used by the app
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

**2. (Recommended) create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
python -m streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

### ⚠️ Troubleshooting: `Can't get attribute '_RemainderColsList'`

The saved models were trained with **scikit-learn 1.6.1**. Loading them with a different version can fail. Install the exact version:

```bash
python -m pip install scikit-learn==1.6.1
```

### Re-training the models

Open `notebooks/House_Price_Prediction.ipynb` (originally run in Google Colab), run all cells, and save the new pipelines into `models/`. For the notebook you will also need `matplotlib`, `seaborn` and `jupyter`.

---

## ⚠️ Limitations

- Data is from the **1990 census**, so predictions reflect historical prices, not today's market.
- Each row is a **block group** (a neighbourhood), not an individual house, so results are neighbourhood-level estimates.
- Prices are **capped at $500,001**, so very expensive areas are under-predicted.
- Only two models were compared, and only simple ratio features were engineered.
- Predictions are statistical estimates and **not property valuations**.

---

## 🔭 Future Improvements

- Try ensemble models such as Random Forest, Gradient Boosting, XGBoost or LightGBM
- Add more engineered features (e.g. distance to major cities or the coast)
- Add model explainability with SHAP
- Use more recent housing data
- Deploy the app online with Streamlit Community Cloud

---

## 👤 Author

**Pulusu Venkata Naga Varshith**
B.Tech, Computer Science and Engineering — Amrita Vishwa Vidyapeetham, Chennai

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: *add your link*
- Email: *add your email*

---

## 📄 License

This project is released under the [MIT License](LICENSE).

⭐ If you found this project useful, consider giving the repository a star!
