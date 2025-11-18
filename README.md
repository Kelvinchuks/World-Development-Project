# World Development Data Project

## Overview
This project explores and classifies countries based on their socio-economic development indicators using machine learning methods: clustering, regression, and classification, with data from the World Bank (1985–2023).

## Objectives
- **Clustering**: Identify groups of countries with similar economic/environmental profiles.
- **Regression**: Examine predictors of inflation volatility, fertility, female labour participation, and growth.
- **Classification**: Predict high-inequality, economic instability, income group, and emissions categories.

## Dataset
Source: [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators)

Schema:
- GDP per capita
- Internet Use
- FDI
- CO₂ Emissions
- Health Expenditure
- And more (23 indicators total, 1985–2023)

## ⚙️ Methods and Tools

### 🔹 Data Processing
- `pandas`, `numpy`, `KNNImputer`, `MinMaxScaler`

###  Clustering
- `KMeans`
- Validation: Elbow Method, Silhouette Score, Davies–Bouldin Index

###  Regression
- `OLS`, `Panel Regression`, `LassoCV`
- Focus: Inflation volatility, labor force participation, fertility trends

###  Classification
- `LogisticRegression`, `RandomForestClassifier`
- ROC Curve, AUC, GridSearchCV

###  Visualization
- `matplotlib`, `seaborn`, `PCA`, correlation heatmaps, time series plots

## Project Structure

- `notebooks/` - Final Jupyter notebook with the full pipeline
- `src/` - Modular Python scripts for reproducibility
- `data/` - Raw and cleaned datasets
- `visuals/` - EDA and model visuals
- `reports/` → findings and summaries, research references

##  Key Findings
- Internet access and education strongly correlate with lower fertility
- Countries cluster along lines of trade openness and digital infrastructure
- Renewable energy negatively correlates with emissions, but not income

##  Insights
- Regression and clustering both reveal meaningful regional and development patterns.
- Time-series analysis shows steady fertility decline, internet growth, and fluctuating GDP.

##  References
See [World Development Project](https://github.com/Kelvinchuks/World-Development-Project/blob/Kelvinchuks/World%20Development%20Project.ipynb).

Acknowledgement: Group Project (My Team and I)
