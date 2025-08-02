#!/usr/bin/env python
# coding: utf-8

# # World Development Data Project (Clustering, Regression, Classification)

# ## Case Description
# 
# This project is aiming to explore and classify countries based on their socio-economic development indicators while using supervised learning models. We want to make use of the cleaned dataset derived from the World Bank's development indicators, and in the following, perform various tasks to answer the relevant questions in three main areas (clustering, regression, and classification).
# 

# ## Motivation
# 
# As we are an international group of students - and due some economics studies - we were inspired to look at the whole world and see what trends we can see, what lessons we can learn about the economic and environmental standings of various countries: what are the commonalities and what separates them, and might we gain some insight from our history and current standing on how we can further streer ourselves in the right direction when it comes to the economy, scoiety and the enviroment. To this end, we will analyize the socioeconomic and environment-related metrics found in our dataset as outlined in the following sections.

# ## Dataset Description
# 
# ### Data Source: 
# World Bank Group DataBank: World Development Indicators (https://databank.worldbank.org/source/world-development-indicators)
# 
# Our starting data set contains the 23 selected indicators for all countries for the years 1985 to 2023.
# The indicators ("Series Name" in the dataset) loaded are as follows:
# 
# - GDP per capita, PPP (current international dollar)
# - GDP growth (annual %)
# - Gross fixed capital formation (% of GDP)
# - Foreign direct investment, net inflows (% of GDP)
# - Exports of goods and services (% of GDP)
# - Imports of goods and services (% of GDP)
# - Inflation, consumer prices (annual %)
# - Unemployment, total (% of total labor force) (modeled ILO estimate)
# - Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)
# - Gini
# - Life expectancy at birth, total (years)
# - Fertility rate, total (births per woman)
# - School enrollment, tertiary (% gross)
# - School enrollment, tertiary, female (% gross)
# - Current health expenditure per capita (current US$)
# - Current health expenditure (% of GDP)
# - Individuals using the Internet (% of population)
# - Mobile cellular subscriptions (per 100 people)
# - Fixed broadband subscriptions (per 100 people)
# - Carbon dioxide (CO2) emissions (total) excluding LULUCF (% change from 1990)
# - Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
# - Renewable energy consumption (% of total final energy consumption)
# - Electric power consumption (kWh per capita)
# 
# 
# The loaded dataset contains the following columns:
# - Country Name
# - Country Code
# - Series Name
# - Series Code [a unique idetified for the series]
# - A column for each year

# ## The questions addressed in this notebook are:
# 
# **Clustering**
# 
# - Question 1: How can we discover hidden socioeconomic development based on trade openness (exports, imports), internet/mobile penetration, and tertiary enrolment?
# - Question 2: Do countries cluster into distinct groups based on environmental and energy indicators (CO₂ emissions levels/trends, renewable energy % of use, and per capita energy consumption) along with economic size?
# - Question 3: How have clusters changed over time? (revisiting the topics discussed in Questions 1 and 2)
# 
# **Regression**
# 
# - Question 1: What economic or social variables best predict lower inflation volatility across nations?
# - Question 2: To what extent can internet access explain differences in fertility rates, controlling for female labour-force participation and education?
# - Question 3: What factors predict the female labor force participation rate?
# - Question 4: Do more open economies (with higher trade and investment relative to GDP) experience faster growth? 
# 
# **Classification**
# 
# - Question 1: Which combination of development indicators can classify a country as having high inequality (GINI index above threshold)?
# - Question 2: Classification of countries that are at risk of economic instability based on early warning signals e.g rising inflation, declining exports, and infant mortality?
# - Question 3: Use of development indicators to classify countries as High-Income or not. What are the defining features of high-income countries?
# - Question 4: Can we classify countries as high CO₂ emitters versus low emitters based on their development indicators?

# ## Related Work
# 
# ### Clustering:
# 
# This section uses unsupervised learning to uncover potential hidden groupings in the socio-economic and environment-related data. We apply distance-based (K-Means) clustering to scaled World Bank Development Indicators, using multiple validation metrics to ensure robust segmentation.
# 
# **Methodologies**
# - K-Means Clustering* 
#   – Classic Lloyd/Hartigan-Wong implementation, optimized via scikit-learn’s `KMeans`.  
# - Cluster Validation 
#   – Elbow (WSS) to locate the inertia “knee.”  
#   – Silhouette Index (Rousseeuw, 1987) for average cluster separation.  
#   – Davies–Bouldin Score for intra- vs. inter-cluster ratios.  
# - Dimensionality Reduction  
#   – Principal Component Analysis (PCA) to visualize clusters in 2D.  
# 
# **Tools and Libraries**
# - Data Processing: `pandas`, `numpy`  
# - Clustering & Validation: `scikit-learn` (`KMeans`, `silhouette_score`, `davies_bouldin_score`)  
# - Visualization: `matplotlib`, `seaborn`  
# 
# **References**
# - Hartigan, J. A., & Wong, M. A. (1979). Algorithm AS 136: A k-means clustering algorithm. _Journal of the Royal Statistical Society, Series C_
#   
# - Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. _Journal of Computational and Applied Mathematics_
#    
# - Kaufman, L., & Rousseeuw, P. J. (1990). _Finding Groups in Data: An Introduction to Cluster Analysis_. Wiley
#    
# - World Bank (2024). World Development Indicators.
#   – Annual, country-level metrics (1990–2023) on trade openness, digital penetration, tertiary enrollment, CO₂ emissions, renewable energy share, per-capita power use, and GDP per capita.
#   
# -  World Bank. (n.d.). *Access to sustainable energy*. Retrieved June 29, 2025, from https://www.worldbank.org/en/cpf/india/what-we-work/resource-efficient-growth/access-sustainable-energy
#   
# - World Bank. (n.d.). *Energy overview*. Retrieved June 29, 2025, from https://www.worldbank.org/en/topic/energy/overview
# 
# - World Bank. (2018, April 3). *Stronger open trade policies enable economic growth for all*. https://www.worldbank.org/en/results/2018/04/03/stronger-open-trade-policies-enables-economic-growth-for-all
#   
# - World Bank. (n.d.). *Trade overview*. Retrieved June 29, 2025, from https://www.worldbank.org/en/topic/trade/overview
#   
# - World Bank. (2024). Sustainability Review 2023. © World Bank. http://hdl.handle.net/10986/41150 License: CC BY-NC 3.0 IGO.
# 
# ### Regression:
# 
# This project builds on a growing body of interdisciplinary research at the intersection of development economics, gender studies, macroeconomics, and digital transformation. The regression questions focus on understand relationships between global development indicators and macro-social outcomes such as inflation volatility, fertility, labor force participation, and economic growth and tested using panel, Lasso and OLS models
# 
# **Inflation Volatility and Predictors of Macroeconomic Stability**
# 
# Inflation volatility remains a core concern in macroeconomic stability studies. Multiple empirical analyses have established that income levels, inequality (GINI), digital access, and public health spending are significant correlates of price stability across countries.
# 
# - For instance, Cukierman et al. (1992) link institutional quality and income levels with inflation performance.
# 
# - Recent literature also explores how fiscal discipline, technological diffusion, and health system robustness contribute to lower inflation uncertainty (IMF, 2021).
# 
# - GINI’s role is especially important in inflation dynamics, as inequality often correlates with monetary policy credibility and transmission asymmetries (Ayyagari et al., 2019).
# 
# **Sources**:
# 
# - Cukierman, A., Edwards, S., & Tabellini, G. (1992). Seigniorage and political instability. American Economic Review.
# 
# - IMF. (2021). World Economic Outlook: Managing Divergent Recoveries.
# 
# - Ayyagari, M., Demirgüç-Kunt, A., & Maksimovic, V. (2019). Who creates jobs in developing countries? Small Business Economics.
# 
# **Internet Access, Fertility, and Female Agency**
# 
# The impact of internet diffusion on fertility choices and female labor outcomes has been increasingly studied in the last decade, especially in the context of digital inclusion, gender empowerment, and access to information.
# 
# - Billari et al. (2019) found that digital access is associated with delayed childbearing and lower fertility, particularly when controlling for education and workforce engagement.
# 
# - UNESCO and the World Bank have emphasized how internet access modifies the effects of higher education by offering women more flexible life paths and exposure to family planning tools.
# 
# - However, some studies caution that internet access alone is not deterministic, and cultural, educational, and labor market conditions mediate its effects (Graham & Dutton, 2014).
# 
# **Sources:**
# 
# - Billari, F. C., Giuntella, O., & Stella, L. (2019). Internet access and fertility: Evidence from a natural experiment. Review of Economics and Statistics.
# 
# - World Bank. (2022). Digital Development Overview.
# 
# - Graham, M., & Dutton, W. H. (2014). Society and the Internet. Oxford University Press.
# 
# **Determinants of Female Labor Force Participation**
# 
# Female Labor Force Participation (FLFP) is widely studied in development economics. Core determinants include education, internet access, health expenditure, and income inequality.
# 
# - Klasen & Pieters (2015) observed that education increases FLFP over the long run, although in some contexts, women in tertiary education may temporarily withdraw from the labor market during their studies.
# 
# - Internet access has been found to enhance FLFP by increasing access to job markets, enabling remote work, and reducing search frictions (ITU, 2020).
# 
# - High inequality (GINI) often constrains women’s access to employment opportunities and correlates with weaker labor protections (OECD, 2018).
# 
# **Sources:**
# 
# - Klasen, S., & Pieters, J. (2015). What explains the stagnation of female labor force participation in urban India?. World Bank Economic Review.
# 
# - ITU (2020). Measuring Digital Development: Facts and Figures.
# 
# - OECD (2018). Bridging the Digital Gender Divide.
# 
# **Trade Openness, FDI, and Growth Performance**
# 
# The relationship between openness to trade and investment and GDP growth is a longstanding topic in international economics. While theory predicts a positive link, empirical evidence is mixed.
# 
# - Sachs and Warner (1995) find that open economies tend to grow faster, particularly when institutions support trade and capital flows.
# 
# - However, more recent work emphasizes that FDI is only growth-enhancing in countries with high institutional quality and absorptive capacity (Borensztein et al., 1998).
# 
# - Panel studies suggest that trade openness has modest lagged effects on growth, while FDI may have no effect unless combined with human capital (Alfaro et al., 2004).
# 
# **Sources:**
# 
# - Sachs, J., & Warner, A. (1995). Economic reform and the process of global integration. Brookings Papers.
# 
# - Borensztein, E., De Gregorio, J., & Lee, J. W. (1998). How does foreign direct investment affect economic growth?. Journal of International Economics.
# 
# - Alfaro, L., Chanda, A., Kalemli-Ozcan, S., & Sayek, S. (2004). FDI and economic growth: The role of local financial markets. Journal of International Economics.
# 
# **Tools and Methodologies**
# 
# - Pandas for data processing and manipulation.
# 
# - Statsmodels and linearmodels for OLS and panel regression analysis.
# 
# - Scikit-learn (LassoCV) for regularized regression and feature selection.
# 
# - Jupyter Notebooks for reproducible and interpretable analysis.
# 
# ### Classification:
# 
# This project builds upon wide literature and practical applications in socio-economic classification using machine learning. The analysis as well builds on several well-established work in the fields of development economics, data science, and machine learning. Our aim was to understand how global development indicators relate to outcomes such as income classification, economic inequality, environmental emissions, and macroeconomic stability.
# 
# **Development Indicators & Classification**
# 
# The use of development indicators such as GDP per capita, education levels, health expenditures, and fertility rates to classify countries is widely supported in the literature.
# E.g.: World Bank Development Reports, UNDP Human Development Index documentation. Previous empirical studies have shown a strong correlation between these indicators and outcomes like inequality (e.g., GINI), economic instability, and environmental impact (CO₂ emissions). Reference: Kuznets (1955) on inequality–income curve, and more recent cross-country analyses by UNDP and OECD.
# 
# **Machine Learning Approaches in Development Research**
# 
# Machine learning techniques have been frequently applied in global development work, including:
# 
# - Predictive modeling of economic risks using Random Forests and Logistic Regression (cf. Goldstein et al., 2019).
# 
# - Income group classification from socio-economic and infrastructure data (OpenAI, World Bank ML sandbox projects).
# 
# - The World Bank Data Science Lab has provided notebooks and workshops that guide the use of scikit-learn and pandas for such classifications.
# 
# - E.g.: "World Bank Indicators ML"
# 
# **Additional Techniques and Interpretability Tools**
# 
# To enhance the robustness, transparency, and interpretability of our analysis, the following advanced machine learning evaluation tools were incorporated:
# 
# **ROC Curves & AUC**
# 
# ROC curves and Area Under Curve (AUC) are standard metrics for evaluating binary classification performance beyond accuracy.
# 
# **Applied using:**
# 
# - sklearn.metrics.roc_curve and roc_auc_score
#     
# - Inspired by materials from Andrew Ng’s ML course (Coursera) and StatQuest by Josh Starmer.
# 
# **Decision Tree Visualization**
# 
# - Used sklearn.tree.plot_tree to visually interpret decision paths.
# 
# - It is helpful in identifying the most informative features and presenting transparent decision rules to non-technical stakeholders.
# 
# Inspired by interpretability methods from:
# 
# - Ribeiro et al. (2016) “Why Should I Trust You?” – LIME, and
#     
# - Shapley value frameworks for tree-based models (Lundberg et al., 2018).
# 
# - Hyperparameter Tuning (GridSearchCV)
# 
# - Implemented GridSearchCV for optimizing classifier parameters.
# 
# **Based on:**
# 
# - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron (2nd ed).
# 
# - Best practices shared in Scikit-learn documentation and notebooks from Kaggle competitions.
# 
# **Tools and Libraries Used**
# 
# - Pandas for data manipulation
# 
# - Scikit-learn for classification models, hyperparameter tuning, ROC, and plotting
# 
# - Matplotlib for visualizations
# 
# - Jupyter Notebook environment for reproducible analysis
# 

# In[4]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Step 1: We Load raw data

# In[5]:


df_raw = pd.read_excel('world_development_data.xlsx')
df_raw.head(5)


# In[6]:


# Now, we replace the ".." symbol with NaN so we can further can further check missing values:
#.replace(A, B, inplace = True) to replace A by B.

df_raw.replace('..',np.NaN, inplace = True)
df_raw.head(5)


# In[7]:


# Evaluating for Missing Data
# The missing values are converted by default. We use the following functions to identify these missing values.

missing_data = df_raw.isnull()
missing_data.head(5)


# In[8]:


# Count missing values in each column
# We use a for loop in Python to quickly figure out the number of missing values in each column. As mentioned above, "True" represents a missing value and "False" means the value is present in the data set. In the body of the for loop the method ".value_counts()" counts the number of "True" values.

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# Step 2: Melt data to long format

# In[9]:


# We use this code to reshapes the dataset from a wide year-column format into a long format where:
# Each row is a single observation for a country, a development indicator, and a specific year.

id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
df_melted = df_raw.melt(id_vars=id_vars, var_name='Year', value_name='Value')
df_melted['Year'] = df_melted['Year'].str.extract(r'(\d{4})').astype(float)
df_melted.head(10)


# Step 3: Pivot data to wide format

# In[10]:


# We use this step reshapes the data so that:
# Each row is a unique combination of a country and year
# Each column is a different indicator (like GDP, CO₂ emissions)

df_pivot = df_melted.pivot_table(
    index=['Country Name', 'Country Code', 'Year'],
    columns='Series Name',
    values='Value',
    aggfunc='first'  # Only one value per group
).reset_index()
df_pivot.head(5)


# Step 4: Rename important columns for easier access

# In[11]:


col_map = {
    'GDP per capita, PPP (current international $)': 'GDP_pc',
    'GDP growth (annual %)': 'GDP_growth',
    'Gross fixed capital formation (% of GDP)': 'Capital_Formation',
    'Foreign direct investment, net inflows (% of GDP)': 'FDI',
    'Exports of goods and services (% of GDP)': 'Exports',
    'Imports of goods and services (% of GDP)': 'Imports',
    'Inflation, consumer prices (annual %)': 'Inflation',
    'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'Unemployment',
    'Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)': 'Female_LFP',
    'Gini': 'GINI',
    'Life expectancy at birth, total (years)': 'Life_Expectancy',
    'Fertility rate, total (births per woman)': 'Fertility_Rate',
    'School enrollment, tertiary (% gross)': 'Tertiary_Enroll',
    'Current health expenditure per capita (current US$)': 'Health_Exp_pc',
    'Individuals using the Internet (% of population)': 'Internet_Use',
    'Mobile cellular subscriptions (per 100 people)': 'Mobile_Use',
    'Fixed broadband subscriptions (per 100 people)': 'Broadband_Use',
    'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)': 'CO2_Emissions',
    'Renewable energy consumption (% of total final energy consumption)': 'Renew_Energy',
    'Electric power consumption (kWh per capita)': 'Power_Use'
}
df_pivot.rename(columns=col_map, inplace=True)


# Step 5: Filter to columns of interest

# In[12]:


features = list(col_map.values())
df_filtered = df_pivot[['Country Name', 'Country Code', 'Year'] + features]
df_filtered.head(10)


# Step 6: Drop rows with excessive missing values

# In[13]:


# We use this step removes rows (country-year combinations) that have too many missing values in the selected features.

df_filtered = df_filtered[df_filtered[features].isnull().mean(axis=1) < 0.4]


# Step 7: Interpolate missing values

# In[14]:


# We use this code fills in missing values in our dataset using linear interpolation, separately for each country, across years. 
# It's a smart and domain-appropriate method for time-series–like data such as country development indicators.

df_filtered[features] = df_filtered.groupby(['Country Code'])[features].transform(lambda group: group.interpolate(limit_direction='both'))


# Step 8: KNN Imputation for remaining missing values

# In[15]:


# We fill in any remaining missing values using a machine learning-based technique called K-Nearest Neighbors (KNN) Imputation.
imputer = KNNImputer(n_neighbors=5)
df_filtered[features] = imputer.fit_transform(df_filtered[features])


# Step 9: Feature Engineering

# In[16]:


# We use this code to create new, meaningful features from existing data to improve the quality and predictive power of economic or development analysis

# GDP volatility with min 2 periods
df_filtered['GDP_volatility'] = df_filtered.groupby('Country Code')['GDP_growth'].transform(
    lambda x: x.rolling(window=5, min_periods=2).std()
)
df_filtered['GDP_volatility'] = df_filtered['GDP_volatility'].fillna(df_filtered['GDP_volatility'].mean())

# Lagged internet use — fill first row per country
df_filtered['Internet_Use_Lag1'] = df_filtered.groupby('Country Code')['Internet_Use'].shift(1)
df_filtered['Internet_Use_Lag1'] = df_filtered['Internet_Use_Lag1'].fillna(df_filtered['Internet_Use_Lag1'].mean())

# Fertility category — dynamic bin
df_filtered['Fertility_Category'] = pd.cut(
    df_filtered['Fertility_Rate'],
    bins=[0, 2, 4, df_filtered['Fertility_Rate'].max()],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)


# Step 10: Normalize features

# In[17]:


# We us this to normalize the selected numerical features so they all lie in the same scale (typically 0 to 1). This is important for machine learning models
scaler = MinMaxScaler()
df_scaled = df_filtered.copy()
df_scaled[features] = scaler.fit_transform(df_scaled[features])


# In[18]:


# We use this to check if all datatype are set to standard
df_filtered.dtypes


# In[19]:


# We converted the Year datatype to int
df_filtered['Year'] = df_filtered['Year'].astype(int)
df_scaled['Year'] = df_filtered['Year'].astype(int)


# Since most of our data is missing, we narrow it down from 1990 - 2023

# In[20]:


import pandas as pd

# This ensure 'Year' is still integer
df_filtered['Year'] = df_filtered['Year'].astype(int)

# Define the required year range (1990 to 2023 inclusive)
required_years = set(range(1990, 2024))  # 2024 is excluded → goes up to 2023

# Find countries that cover all required years
country_years = df_filtered.groupby(['Country Name', 'Country Code'])['Year'].apply(set).reset_index()
country_years['Has_All_Years'] = country_years['Year'].apply(lambda years: required_years.issubset(years))

# Get only those countries
full_range_countries = country_years[country_years['Has_All_Years'] == True][['Country Name', 'Country Code']]

# Filter the full dataset for these countries
df_filtered_1990_2023 = df_filtered.merge(full_range_countries, on=['Country Name', 'Country Code'])

# Keep only the desired columns (including all engineered and economic indicators)
selected_columns = [
    'Country Name', 'Country Code', 'Year',
    'GDP_pc', 'GDP_growth', 'Capital_Formation', 'FDI', 'Exports', 'Imports',
    'Inflation', 'Unemployment', 'Female_LFP', 'GINI', 'Life_Expectancy', 'Fertility_Rate',
    'Tertiary_Enroll', 'Health_Exp_pc', 'Internet_Use', 'Mobile_Use', 'Broadband_Use',
    'CO2_Emissions', 'Renew_Energy', 'Power_Use',
    'GDP_volatility', 'Internet_Use_Lag1', 'Fertility_Category'
]

# Final clean dataset
df_filtered_1990_2023 = df_filtered_1990_2023[selected_columns]
df_filtered_1990_2023.head(10)


# In[21]:


import pandas as pd

# This ensure 'Year' is still integer
df_scaled['Year'] = df_scaled['Year'].astype(int)

# Define the required year range (1990 to 2023 inclusive)
required_years = set(range(1990, 2024))  # 2024 is excluded → goes up to 2023

# Find countries that cover all required years
country_years = df_scaled.groupby(['Country Name', 'Country Code'])['Year'].apply(set).reset_index()
country_years['Has_All_Years'] = country_years['Year'].apply(lambda years: required_years.issubset(years))

# Get only those countries
full_range_countries = country_years[country_years['Has_All_Years'] == True][['Country Name', 'Country Code']]

# Filter the full dataset for these countries
df_scaled_1990_2023 = df_scaled.merge(full_range_countries, on=['Country Name', 'Country Code'])

# Keep only the desired columns (including all engineered and economic indicators)
selected_columns = [
    'Country Name', 'Country Code', 'Year',
    'GDP_pc', 'GDP_growth', 'Capital_Formation', 'FDI', 'Exports', 'Imports',
    'Inflation', 'Unemployment', 'Female_LFP', 'GINI', 'Life_Expectancy', 'Fertility_Rate',
    'Tertiary_Enroll', 'Health_Exp_pc', 'Internet_Use', 'Mobile_Use', 'Broadband_Use',
    'CO2_Emissions', 'Renew_Energy', 'Power_Use',
    'GDP_volatility', 'Internet_Use_Lag1', 'Fertility_Category'
]

# Final clean dataset
df_scaled_1990_2023 = df_scaled_1990_2023[selected_columns]
df_scaled_1990_2023.head(10)


# Step 11: Export cleaned data

# In[22]:


# df_filtered_1990_2023.to_csv('clean_world_dev_data.csv', index=False)
# df_scaled_1990_2023.to_csv('scaled_world_dev_data.csv', index=False)


# In[23]:


# print("Data cleaning and transformation complete. Files saved:")
# print("- clean_world_dev_data.csv")
# print("- scaled_world_dev_data.csv")


# ## Exploratory Data Analysis (EDA)

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetics
sns.set(style="whitegrid")


# In[25]:


# Summary statistics
print(df_filtered_1990_2023.describe())


# **Summary Statistics Interpretation**
# 
# This dataset captures a wide range of macroeconomic, social, technological, and environmental variables. Below is an insight-by-variable interpretation.
# 
# **Macroeconomic Indicators**
# 
# | **Variable**                        | **Interpretation**                                                                                                                    |
# | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
# | `GDP_pc` (Mean = \$18,025)          | Very large global range (min = \$311, max = \$149,794) with high std = \$20,613. Indicates strong income inequality across countries. |
# | `GDP_growth` (Mean = 3.36%)         | Wide variation from deep recessions (–54%) to booms (+75%). Highlights periods of instability and recovery.                           |
# | `Capital_Formation` (Mean = 22.36%) | Reasonable investment levels globally, but some countries experienced contraction (min = –2.42%).    |
# 
# **Trade and Investment**
# 
# | Variable                         | Insight                                                                                                                                                                            |
# | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **FDI (Mean = 4.96%)**           | Foreign Direct Investment is skewed: **std = 24**, with some extreme inflows (**max = 452%**) and outflows (**min = –440%**). May include one-time spikes or accounting anomalies. |
# | **Exports & Imports (% of GDP)** | Economies range from **very closed (\~4%)** to **extremely open (over 220%)**, indicating both self-reliant and highly globalized nations.                                         |
# 
# **Stability Metrics**
# 
# | Variable                          | Insight                                                                                                                           |
# | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
# | **Inflation (Mean = 19.4%)**      | Extremely high **std = 194.6** and **max = 7,481%** show the presence of **hyperinflation episodes** (e.g., Venezuela, Zimbabwe). |
# | **Unemployment (Mean = 7.44%)**   | Reasonable global average, but wide spread (**max = 38.8%**) hints at systemic joblessness in some nations or years.              |
# | **GDP\_volatility (Mean = 2.96)** | Indicates average annual volatility in GDP growth; extreme value (**max = 48.6**) implies deep instability in some economies.     |
# 
# **Gender, Education & Labor**
# 
# | Variable                             | Insight                                                                                                                                |
# | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
# | **Female\_LFP (Mean = 50%)**         | Substantial gender variance in labor participation (**min = 9.7%, max = 90.5%**), likely reflecting cultural and economic constraints. |
# | **Tertiary\_Enroll (Mean = 33.8%)**  | Moderate global enrollment with wide disparities (**max = 167%**, due to double-counting or adult education).                          |
# | **Health\_Exp\_pc (Mean ≈ \$1,089)** | Ranges from very low (**min = \$5.68**) to very high (**max = \$12,434**), reflecting inequality in healthcare access.                 |
# 
# **Technology & Digital Access**
# 
# | Variable                         | Insight                                                                                                                                |
# | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use (Mean = 29.7%)** | The global average is still low compared to today's standards. Large spread (**std = 32.9**, **max = 100%**) shows the digital divide. |
# | **Mobile\_Use (Mean = 62.2%)**   | Higher than internet, showing wider global access to mobile phones. **Max exceeds 400%** due to multiple subscriptions per person.     |
# | **Broadband\_Use (Mean = 7.7%)** | Very low overall; many countries may lack broadband infrastructure.                                                                   |
# **Environment & Energy**
# 
# | Variable                                 | Insight                                                                                                                           |
# | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
# | **CO₂ Emissions (Mean = 225 tons)**      | Highly skewed; **max = 13,260** shows big emitters (e.g., U.S., China); **min = 0.04** includes very low-emission nations.       |
# | **Renewable Energy (% of total)**        | Wide spread (**mean = 34.2%**, **range: 0–97.3%**) implies some countries rely heavily on renewables, others not at all.          |
# | **Power\_Use (Mean = 3,751 kWh/capita)** | Reflects economic activity; extreme **max = 55,085** shows outliers like industrialized nations or small energy-exporting states. |
# 
# **Lagged Variables**
# 
# | Variable                               | Insight                                                                                                                                                |
# | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | **Internet\_Use\_Lag1 (Mean = 28.3%)** | Suggests a roughly 1-year delay in the diffusion of digital technologies, used in models to test delayed effects on outcomes like fertility or growth. |
# 

# **Correlation heatmap**

# In[26]:


plt.figure(figsize=(16, 10))
sns.heatmap(df_filtered_1990_2023[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Development Indicators")
plt.show()


# **Interpretation of Correlation Between Development Indicators**
# 
# The correlation matrix reveals the strength and direction of linear relationships between global development indicators (from –1 to +1). Red = strong positive, blue = strong negative.
# 
# **Economic Indicators**
# 
# | Pair                           | Correlation                                        | Interpretation                                                                                                                                         |
# | ------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | **GDP\_pc <-> Exports**          | **0.57**                                           | Wealthier countries tend to have larger export shares. Reflects industrial capacity and global integration.                                            |
# | **GDP\_pc <-> Internet\_Use**    | **0.70**                                           | Strong link: richer countries have better internet infrastructure and penetration.                                                                     |
# | **GDP\_pc <-> Health\_Exp\_pc**  | **0.76**                                           | High-income nations spend more on health per capita, consistent with World Bank findings.                                                        |
# | **GDP\_pc <-> Life\_Expectancy** | **0.66**                                           | Higher income correlates with better health and longevity.                                                                                             |
# | **GDP\_pc <-> GINI**             | **–0.43**                                          | Negative correlation: richer countries tend to have **lower income inequality**.                                                                       |
# | **GDP\_growth <-> Others**       | Generally **weak correlations** (e.g., max \~0.15) | Implies **growth rate is not tightly linked** to most structural indicators. Likely influenced by volatility, cycles, or shocks. |
# 
# **Gender, Labor, and Fertility**
# 
# | Pair                                   | Correlation       | Interpretation                                                                                                                                         |
# | -------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | **Female\_LFP <-> GINI**                 | **–0.42**         | Higher inequality is associated with lower female labor force participation.                                                                           |
# | **Fertility\_Rate <-> Internet\_Use**    | **–0.70**         | Strong negative correlation: higher internet access is associated with **lower fertility**, possibly due to improved information access and education. |
# | **Fertility\_Rate <-> Tertiary\_Enroll** | **–0.70**         | More educated populations tend to have fewer children - consistent with demographic transition theory.                                                 |
# | **Female\_LFP <-> GDP\_pc**              | **+0.081** (weak) | Slightly positive, but too weak to suggest direct linear dependence.                                                                                   |
# 
# **Technology and Connectivity**
# 
# | Pair                                 | Correlation | Interpretation                                                                     |
# | ------------------------------------ | ----------- | ---------------------------------------------------------------------------------- |
# | **Internet\_Use <-> Tertiary\_Enroll** | **0.74**    | Suggests **digital inclusion and higher education go hand-in-hand**.               |
# | **Broadband\_Use <-> Internet\_Use**   | **0.82**    | Logical; broadband boosts internet quality and access.                            |
# | **Mobile\_Use <-> Internet\_Use**      | **0.66**    | High correlation, but slightly lower - not all mobile use implies internet access. |
# 
# **Energy & Emissions**
# 
# | Pair                               | Correlation | Interpretation                                                                                                                             |
# | ---------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
# | **CO2\_Emissions <-> Power\_Use**    | **0.61**    | High power consumption aligns with higher emissions; expected in industrialized nations.                                                  |
# | **Renew\_Energy <-> CO2\_Emissions** | **–0.26**   | Modest negative link: greater renewable share correlates with lower emissions.                                                             |
# | **Renew\_Energy <-> GDP\_pc**        | **–0.26**   | In some richer countries, renewables are low may be due to legacy fossil infrastructure. In others, they’re high - explains moderate correlation. |
# 
# 

# **Time Series Trend**

# In[27]:


df_time_avg = df_filtered_1990_2023.groupby('Year')[
    ['Internet_Use', 'Fertility_Rate', 'Female_LFP', 'GDP_growth']
].mean().reset_index()

df_time_avg.plot(x='Year', figsize=(12, 6), title="Global Averages Over Time")
plt.ylabel("Average Value")
plt.grid(True)
plt.tight_layout()
plt.show()


# **Time Trend Interpretation: Global Averages Over Time (1990–2023)**
# 
# This line chart shows the average yearly trend (across all countries) of:
# 
# **Internet_Use (Blue Line) – Exponential Growth**
# 
# - Rapid and consistent rise from near-zero in 1990 to over 70% by 2023 globally.
# 
# - Reflects global digital transformation, mobile tech diffusion, and infrastructure investments.
# 
# - Most significant change among the indicators, suggesting its strong influence on modern development trends.
# 
# **Implication:** Internet penetration likely plays a key role in driving outcomes such as lower fertility, higher education access, and labor market changes.
# 
# **Female_LFP (Green Line) – Gradual Increase**
# 
# - Began around 47% in 1990, slowly increased to just over 52% by 2023.
# 
# - Shows modest global progress in female labor inclusion over three decades.
# 
# - Slight dip around 2020, likely due to COVID-19’s disproportionate impact on women in labor markets.
# 
# **Implication:** Despite progress, gender gaps in employment persist; stronger policies may be needed to accelerate inclusion.
# 
# **Fertility_Rate (Orange Line) – Steady Decline**
# 
# - Declined from ~3.2 births per woman in 1990 to below 2.5 by 2023.
# 
# **Implication:** Fertility decline aligns with rising education and internet access. This relationship is central to your regression analysis.
# 
# **GDP_growth (Red Line) – Fluctuating & Volatile**
# 
# - Highly cyclical pattern with visible recessions in 2009 and 2020.
# 
# - 2009: Global Financial Crisis
# 
# - 2020: COVID-19 Shock (notably sharp drop into negative territory)
# 
# - Otherwise ranges between 2%–5%, showing some post-shock recoveries.
# 
# **Implication:** Growth trends are volatile and influenced by global crises more than slow-moving development variables.
# 
# 

# **Outlier Detection Using Z-Scores**

# In[28]:


from scipy.stats import zscore

# Compute Z-scores for highly skewed vars
z_scores = df_filtered_1990_2023[['FDI', 'Inflation', 'GDP_growth']].apply(zscore)

# Find rows with z > 3
outliers = (z_scores.abs() > 3).any(axis=1)
print(f"Number of outlier rows: {outliers.sum()}")
df_filtered_1990_2023[outliers].head(95)


# **Note on Outliers**

# For the outliers we carefully considered it based on their nature and impact on the regression model.
# 
# Scaling elimiantes the effect of outliers where use of the scaled dataset is preferably, such as when it comes to clustering.

# ## Clustering

# In this section, we will use regression methods to find similarities within countries based on observations and group them in clusters. For this purpose, we will be using the scaled dataset (a mean of 0 and a unit-variance, i.e., standard deviation of 1)) to ensure that metrics, such as GPD which are measures in the tens of thousands per capita, do not dominate distance-based clusering calculations such as K-Means. As such, all of our variables have the same weight by default.
# 
# Clustering is unsupervised learning, meaning we do not have a target variable, but we rather seek to gain a better understnading of the variables in our data.
# 
# We will aim to answer the following questions:
# 
# - Question 1: How can we discover hidden socioeconomic development based on trade openness (exports, imports), internet/mobile penetration, and tertiary enrolment?
# - Question 2: Do countries cluster into distinct groups based on environmental and energy indicators (CO₂ emissions levels/trends, renewable energy % of use, and per capita energy consumption) along with economic size?
# - Question 3: How have clusters changed over time? (revisiting the topics discussed in Questions 1 and 2)

# ## Question 1: How can we discover hidden socioeconomic development based on trade openness, internet/mobile penetration, and tertiary enrolment?

# One dilemma that arises is how to treat the decaded of time-series data for this analysis. Among the apporaches considered was the analysis of changes over time, e.g. via using the delta of the relevant variables for clustering. While this would show us (via the scaled data) which countries advanced or fell behind others, it would not differentiate between those seemingly "stagnating" countries that consistently remained in the forefront, versus those that ramained in the back - and those in between. Here, scaled data would also fail to show absolute changes over time, only the relative. 
# 
# While such an analysis could be fruitful when combining/cross-referencing the changes with snapshots and the scaled with the unscaled data, for the sake of simplicity, we shall first examine a snapshot of the most recent year, 2023. This naturally biases us to the present and ignores changes over the years (tendencies, trajectories, which countries changed most). Why will be patrially rectified in following sections (e.g. by comparing 2023 to our earliest retained data, 1990).
# 
# First using our 2023 snapsgot data, we shall filter for five indicators 'Exports', 'Imports', 'Internet_Use', 'Mobile_Use', 'Tertiary_Enroll', that is:
# - Exports of goods and services (% of GDP)
# - Imports of goods and services (% of GDP)
# - Individuals using the Internet (% of population)
# - Mobile cellular subscriptions (per 100 people)
# - School enrollment, tertiary (% gross)
#   
# Out data contains 23 indicators (all numeric, hence fit for clustering without adjustment in this regard), and not all of them are directly relevant for the topic as hand (i.e. CO2 emissions). As there is no automated methods what would eliminate these for clustering (no target variable they could predict), the relevant variables were selected manually.

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the 2023 snapshot of scaled data
snapshot = df_scaled_1990_2023[df_scaled_1990_2023['Year'] == df_scaled_1990_2023['Year'].max()]

# Select features of interest
features = snapshot[['Exports', 'Imports', 'Internet_Use', 'Mobile_Use', 'Tertiary_Enroll']]


# Next, we aim to determine the optimal number of clusters to specifiy before running K-Means clustering.
# We shall use all of the following methods, comprare the output for different numbers of clusters, and see if they point in similar directions:
# - Within-sum-of-squares -> Elbow method
# - Silhouette Index
# - Davies-Bouldin Score
# 
# Naturally, we are looking for the optimal trade-off, where out clusters are not too homogenous but our output is complex enough to have specificity in describing the patterns of countries. There is no single objective corrent solution, hence we shall "eyeball" the output of the three methods.

# In[30]:


# Compute inertia (WSS) for k = 1 to 10
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(features)
    inertias.append(km.inertia_)

# Plot the Elbow curve
plt.figure()
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WSS)')
plt.xticks(range(1, 11))
plt.show()

# Compute silhouette scores for k = 2 to 10
sil_scores = []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(features)
    sil_scores.append(silhouette_score(features, labels))

# Plot the Silhouette score curve
plt.figure()
plt.plot(ks, sil_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(ks)
plt.show()

# Compute Davies-Bouldin scores for k = 2 to 10
db_scores = []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(features)
    db_scores.append(davies_bouldin_score(features, labels))

# Plot the Davies-Bouldin score curve
plt.figure()
plt.plot(ks, db_scores, marker='o')
plt.title('Davies-Bouldin Score vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.xticks(ks)
plt.gca().invert_yaxis()
plt.show()


# Interpretation:
# 
# - Elbow (Inertia): continues a substantial reduction through k=4k.
# - Silhouette has a clear local maximum at k=4.
# - Davies–Bouldin also shows a slight local minimum at k=4.
# 
# Whilke the 2 latter methods would prefer 2 clusters, we are looking to add a bit more to our model (otherwise it is like to merely split developed and not-so-developed nations with not middle ground). The methods align in indicating that 4 clusters would be a good fit for the analysis, hence we shall go with k = 4.
# 
# With this, we can go ahead and perform K-Means clustering on the 2023 snapshot of our five indicators. As missing values were previously addressed, they will not be an issue for K-Means.

# In[31]:


# Fit K-Means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
snapshot.loc[:, 'Cluster'] = kmeans.fit_predict(features)

# SDisplay cluster sizes
cluster_sizes = snapshot['Cluster'].value_counts().sort_index().reset_index()
cluster_sizes.columns = ['Cluster', 'Size']
print("Cluster Sizes:")
print(cluster_sizes)

# Display cluster centers (scaled feature means)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
centers['Cluster'] = centers.index
centers = centers[['Cluster'] + list(features.columns)]
print("\nCluster Centers (scaled features):")
print(centers)

# Plot cluster centers for visual comparison
plt.figure(figsize=(8, 5))
centers_plot = centers.set_index('Cluster')
centers_plot.plot(kind='bar')
plt.title('Cluster Centers (scaled features)')
plt.xlabel('Cluster')
plt.ylabel('Feature value (scaled)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA for 2D visualization (only a 2D summary/appproximation of our 5D analysis)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
snapshot.loc[:, 'PC1'] = pca_result[:, 0]
snapshot.loc[:, 'PC2'] = pca_result[:, 1]

plt.figure(figsize=(6, 5))
for cluster in sorted(snapshot['Cluster'].unique()):
    mask = snapshot['Cluster'] == cluster
    plt.scatter(snapshot.loc[mask, 'PC1'], snapshot.loc[mask, 'PC2'], label=f'Cluster {cluster}', alpha=0.7)
plt.title('PCA of Indicators Colored by Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()


# Let us list which countries belong in which clusters. 
# 
# It is to be noted that we do not have continents/regions assigned to the countries. While we could join this data from other sources, we prefer to retain a region-agnostic analysis in this case.

# In[32]:


clustered = snapshot.groupby('Cluster')['Country Name'].apply(list)

for cluster, countries in clustered.items():
    n = len(countries)
    cols = 4
    rows = (n + cols - 1) // cols 
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = c * rows + r
            row.append(countries[idx] if idx < n else "")
        matrix.append(row)

    print(f"Cluster {cluster} ({n} countries):")
    for row in matrix:
        print("  ".join(f"{item:20}" for item in row))
    print()


# ### Interpreting clusters
# 
# Let us see if we can weave coherent a narrative concerning that we see and name the clusters accordingly. 
# 
# Cluster 0 (30 countries) – “Lower-Performers”
# 
# - All five indicators are below average.
# - These economies have yet to really scale up trade, connectivity, or higher-education enrollment.
# 
# Cluster 1 (42 countries) – “Balanced Mid-Performers”
# 
# - Moderate trade openness and good digital adoption, but tertiary enrollment still middling.
# - Emerging economies that have laid digital and trade foundations but need to boost human-capital investment.
# 
# Cluster 2 (34 countries) – “Digital & Educational Leaders”
# 
# - Top-tier Internet and tertiary enrollment, with solid (though not top) trade openness.
# - Nations that have prioritized human capital and connectivity, now poised to deepen trade links.
# 
# Cluster 3 (6 countries) – “High Flyers”
# 
# - Exceptional trade openness combined with elite digital and tertiary-education metrics.
# - The true frontrunners—small advanced economies or city-states that excel on all dimensions.

# ### Conclusion
# 
# While GDP-based rankings place all high-income economies together, our clustering teases out who leads in digital adoption versus who leads in trade or education.
# 
# Policy peer groups:
# 
# - A country in Cluster 2 (Digital & Educational Leader) but mid-rank in GDP might learn best practices in trade from Cluster 3.
# - A Cluster 1 economy could compare its tertiary-enrollment policies with Cluster 2 peers to close the education gap.
# 
# Targeted diagnostics:
# 
# - Cluster 0 nations need broad interventions—trade facilitation, digital infrastructure, and university capacity building.
# - Cluster 1 should shift emphasis toward higher education, and Cluster 2 toward expanding global trade ties.

# ### Hierarchical clustering approach
# 
# While we will be sticking to the K-Means method going forward, to have some methodological variance, we will briefly rerun this exercise using hierarchical clustering (which itself determines the number of clusters) and comprare the output.

# In[35]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

L = linkage(features, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(
    L,
    labels=snapshot['Country Name'].values,
    leaf_rotation=90,
    leaf_font_size=6
)
plt.title("Hierarchical Clustering Dendrogram – 2023 Snapshot")
plt.xlabel("Country")
plt.ylabel("Linkage Distance")
plt.tight_layout()
plt.show()


# The dendrogram shows splits at various levels of the hierarchy. The main split and coloring corresponds to the 2 clusters certain metrics also suggested for out previous analysis, with pooper countries seemingly in the ornage cluster, while the green one is quite diverse, encompassing Indonecia to Nepal to Switzerland (the latter two mountainous!). While the groups can be further split, I believe the four clusters previously arrived at provide a satisfactory overview.

# ## Question 2: Do countries cluster into distinct groups based on environmental and energy indicators along with economic size?

# The 2023 (most recent) snapshot from question 1 will also serve us well for this exercise, thus we will reuse it, selecting different features as appropriate.
# 
# Here we will include indicators 'CO2_Emissions', 'Renew_Energy', 'Power_Use', 'GDP_pc' in our analysis, the latter capturing the size of the economy, the others emissions and energy-related metrics.
# 
# Afted this, we determine the optimal number of clusters using the same methodology as previously.

# In[30]:


# Select environmental, energy, and economic size features
features_env = snapshot[['CO2_Emissions', 'Renew_Energy', 'Power_Use', 'GDP_pc']]

# Compute inertia (WSS) for k = 1 to 10
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(features_env)
    inertias.append(km.inertia_)

# Plot the Elbow curve
plt.figure()
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WSS)')
plt.xticks(range(1, 11))
plt.show()

# Compute silhouette scores for k = 2 to 10
sil_scores = []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(features_env)
    sil_scores.append(silhouette_score(features_env, labels))

# Plot the Silhouette score curve
plt.figure()
plt.plot(ks, sil_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(ks)
plt.show()

# Compute Davies-Bouldin scores for k = 2 to 10
db_scores = []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(features_env)
    db_scores.append(davies_bouldin_score(features_env, labels))

# Plot the Davies-Bouldin score curve
plt.figure()
plt.plot(ks, db_scores, marker='o')
plt.title('Davies-Bouldin Score vs. Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.xticks(ks)
plt.gca().invert_yaxis() 
plt.show()


# While it is  less clear than previously, k=4 is still a potential sweetspot for the number of clusters.
# 
# - Elbow point: around k=3–4
# - Silhouette peaks at k=2, but the next local maximum is at k=4
# - Davies–Bouldin: while the score decreases beyond 4 clusters (most sharply at 6), 4 remains acceptable given the other methods as not to bloat the number of clusters

# In[31]:


# Fit K-Means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
snapshot.loc[:, 'Cluster'] = kmeans.fit_predict(features_env)

# SDisplay cluster sizes
cluster_sizes = snapshot['Cluster'].value_counts().sort_index().reset_index()
cluster_sizes.columns = ['Cluster', 'Size']
print("Cluster Sizes:")
print(cluster_sizes)

# Display cluster centers (scaled feature means)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_env.columns)
centers['Cluster'] = centers.index
centers = centers[['Cluster'] + list(features_env.columns)]
print("\nCluster Centers (scaled features):")
print(centers)

# Plot cluster centers for visual comparison
plt.figure(figsize=(8, 5))
centers_plot = centers.set_index('Cluster')
centers_plot.plot(kind='bar')
plt.title('Cluster Centers (scaled features)')
plt.xlabel('Cluster')
plt.ylabel('Feature value (scaled)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_env)
snapshot.loc[:, 'PC1'] = pca_result[:, 0]
snapshot.loc[:, 'PC2'] = pca_result[:, 1]

plt.figure(figsize=(6, 5))
for cluster in sorted(snapshot['Cluster'].unique()):
    mask = snapshot['Cluster'] == cluster
    plt.scatter(snapshot.loc[mask, 'PC1'], snapshot.loc[mask, 'PC2'], label=f'Cluster {cluster}', alpha=0.7)
plt.title('PCA of Indicators Colored by Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()


# Once again, let us list the countries per cluster:

# In[32]:


clustered = snapshot.groupby('Cluster')['Country Name'].apply(list)

for cluster, countries in clustered.items():
    n = len(countries)
    cols = 4
    rows = (n + cols - 1) // cols 
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = c * rows + r
            row.append(countries[idx] if idx < n else "")
        matrix.append(row)

    print(f"Cluster {cluster} ({n} countries):")
    for row in matrix:
        print("  ".join(f"{item:20}" for item in row))
    print()


# ### Interpretation: 
# We may designate the clusters as the following:
# 
# Cluster 0: “Mixed Mid-Spectrum”
# 
# - Moderate values on all four axes (CO₂≈0.01, Renew≈0.22, Power≈0.04, GDP≈0.11).
# - (A broad band of middle-income and emerging markets)
# - These countries neither lead nor lag extremely on any one dimension.
# 
# Cluster 1: “Nordic/Renewables Leaders”
# 
# - Very high renewables share (0.64), high power use (0.45) & GDP (0.42), near-zero CO₂.
# - (Countries: Finland, Iceland, Norway, Sweden)
# - These small advanced economies pair large energy consumption with clean grids.
# 
# Cluster 2: “Emerging Renewables”
# 
# - Highest renewables share (0.71) but very low power use (0.01) & GDP (0.04), near-zero CO₂.
# - (Mostly lower-income countries in Africa/Latin America)
# - They rely heavily on traditional renewables (biomass, hydro) but lack overall energy infrastructure.
# 
# Cluster 3: “High Emitters & Consumers”
# 
# - Highest CO₂ emissions (0.06), above-average power use (0.15) & GDP (0.39), low renewables (0.14).
# - (Major advanced economies & large emitters: US, China, Germany, etc.)
# - These are the big industrial/transport hubs with heavy fossil-fuel footprints.
# 
# ### Conclusion:
# 
# The split raptures policy-relevant divides between green leaders, low-income renewables users, fossil-fuel–driven giants, and the broad middle. Each group suggests different priorities—e.g., carbon reduction for Cluster 3, infrastructure build-out for Cluster 2, and scaling clean capacity for Cluster 0.

# ## Question 3: How have clusters changed over time?
# 
# For this point, we shall examine whether the clusters we indetified are consistent and hold up against time. As it is exceedingly challenging to examine a time series year-after-year via clustering and retain a consistent narrative, especially due to the randomness inherent in the k we shall revisit snapshots from the first two question, but instead of looking at our most recent data (2023), we shall look at the oldest data we have retained (1990) and examine how things have hcnged in 33 years. The methodology will remain identical to that previously used with the snapshots for the sake of comparability.
# 
# Is should be noted that the 1990 might be less reliable, more imputation is expected, and collection moethodologies may have changed locally over 33 years especially in developing countries.
# 
# ### Socioeconomic development (Question 1 revisited)

# Retaining k=4, we shall rerun the K-Means clustering from question 1 with indicators from 1990 instead of 2023, and exampine how things changed in 33 years.
# 
# It is to be noted that while retaining four clusters introduces a bias "from the future", and rerunning our previous analyses might point towards a different optimal cluster number for the 1990 data, retaining 4 helps comparability - we shall see if the narratives established fit these new four clusters, or whether they are different beasts entirely.

# In[33]:


# Filter the scaled dataset for 1990
snapshot90 = df_scaled_1990_2023[df_scaled_1990_2023['Year'] == df_scaled_1990_2023['Year'].min()]

# Select features of interest
features = snapshot90[['Exports', 'Imports', 'Internet_Use', 'Mobile_Use', 'Tertiary_Enroll']]

# Fit K-Means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
snapshot90.loc[:, 'Cluster'] = kmeans.fit_predict(features)

# Display cluster sizes
cluster_sizes = snapshot90['Cluster'].value_counts().sort_index().reset_index()
cluster_sizes.columns = ['Cluster', 'Size']
print("Cluster Sizes:")
print(cluster_sizes)

# Display cluster centers/centroids (scaled feature means)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
centers['Cluster'] = centers.index
centers = centers[['Cluster'] + list(features.columns)]
print("\nCluster Centers (scaled features):")
print(centers)

# Plot cluster centers (centroids) for visual comparison
plt.figure(figsize=(8, 5))
centers_plot = centers.set_index('Cluster')
centers_plot.plot(kind='bar')
plt.title('Cluster Centers (scaled features)')
plt.xlabel('Cluster')
plt.ylabel('Feature value (scaled)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
snapshot90.loc[:, 'PC1'] = pca_result[:, 0]
snapshot90.loc[:, 'PC2'] = pca_result[:, 1]

plt.figure(figsize=(6, 5))
for cluster in sorted(snapshot90['Cluster'].unique()):
    mask = snapshot90['Cluster'] == cluster
    plt.scatter(snapshot90.loc[mask, 'PC1'], snapshot90.loc[mask, 'PC2'], label=f'Cluster {cluster}', alpha=0.7)
plt.title('PCA of Indicators Colored by Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()


# In[34]:


clustered = snapshot90.groupby('Cluster')['Country Name'].apply(list)

for cluster, countries in clustered.items():
    n = len(countries)
    cols = 4
    rows = (n + cols - 1) // cols 
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = c * rows + r
            row.append(countries[idx] if idx < n else "")
        matrix.append(row)

    print(f"Cluster {cluster} ({n} countries):")
    for row in matrix:
        print("  ".join(f"{item:20}" for item in row))
    print()


# ### Comparison
# 
# | Archetype                          | 1990 Exports | 1990 Imports | 1990 Internet_Use | 1990 Mobile_Use | 1990 Tertiary_Enroll | Count ’90 | 2023 Exports | 2023 Imports | 2023 Internet_Use | 2023 Mobile_Use | 2023 Tertiary_Enroll | Count ’23 |
# |------------------------------------|-------------:|-------------:|------------------:|----------------:|---------------------:|----------:|-------------:|-------------:|------------------:|----------------:|---------------------:|----------:|
# | **Lowest-Performers**              |        0.086 |        0.118 |             0.000 |          0.000  |               0.041  |        62 |        0.084 |        0.136 |             0.357 |          0.234  |               0.068  |        30 |
# | **Mid-Tier “Tertiary-Smoothers”**  |        0.106 |        0.121 |             0.001 |          0.003  |               0.204  |        34 |        0.161 |        0.188 |             0.805 |          0.298  |               0.266  |        42 |
# | **Trade-Focused**                  |        0.280 |        0.306 |             0.000 |          0.001  |               0.091  |        15 |        0.184 |        0.185 |             0.922 |          0.297  |               0.533  |        34 |
# | **High-Flyers**                    |        0.675 |        0.695 |             0.000 |          0.004  |               0.503  |         1 |        0.583 |        0.552 |             0.949 |          0.388  |               0.472  |         6 |
# 
# 
# - In 1990, Internet and Mobile penetration were effectively zero for everyone. Clusters were driven entirely by trade openness and tertiary‐enrollment levels.
# - By 2023, digital measures have skyrocketed and become core discriminators: mid-tier and trade-focused clusters now show very different Internet/Mobile scores.
# 
# 1990:
# 
# - Hidden patterns were only in trade vs. tertiary education, digital wasn’t yet a factor.
# - A tiny outlier (Singapore) stood far ahead, while most of the world clustered at two or three mid-levels.
# 
# 2023:
# - Digital adoption creates new, clearer separations: you see “Digital & Educational Leaders” distinct from mere trade-focused economies.
# - A broad band of countries has moved into that mid-tier "smooth" cluster, driven by digital expansion.
# - The lowest-performer group has halved, showing widespread catch-up.
# 
# Evolutions of cluster sized
# 
# - Lowest-Performers shrink from 62 to 30 countries: many rpeviously lagging have made enough progress to move out of the bottom rung.
# - Mid-Tier grows from 33 to 42, reflecting a large group that’s both digitally connected and moderately trade‐open, but still building up tertiary enrollment.
# - Trade-Focused “Digital & Educational Leaders” expands from 16 to 34, as dozens of previously trade-only economies have now also invested heavily in higher education and connectivity.
# - High-Flyers jump from 1 to 6: a handful of advanced economies (first Singapore then Hong Kong, Ireland, Luxembourg, Malta, Cyprus) that lead on all five fronts.
# 
# ### Conclusion
# 
# - Digital revolution: Internet and mobile use have gone from negligible to major clustering drivers.
# - Educational catch-up: Many trade‐focused economies added tertiary-enrollment gains, spawning a new cluster of education-and-digital leaders.
# - Shrinking laggards: The bottom group has roughly halved, evidence of broad‐based development since 1990.
# - Policy implication: Benchmark your country’s 1990 vs. 2023 cluster membership to assess which dimension—trade, education, or digital uptake—has been your biggest leap or lingering weakness.

# ### Environmental and energy indicators (Question 2 revisited)
# Once again, we shall retain the cluster size of 4, but use 1990 instead of 2023 data.

# In[35]:


# Filter the scaled dataset for 1990
snapshot90 = df_scaled_1990_2023[df_scaled_1990_2023['Year'] == df_scaled_1990_2023['Year'].min()]

# Select features of interest
features_env = snapshot90[['CO2_Emissions', 'Renew_Energy', 'Power_Use', 'GDP_pc']]

# Fit K-Means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
snapshot90.loc[:, 'Cluster'] = kmeans.fit_predict(features_env)

# Display cluster sizes
cluster_sizes = snapshot90['Cluster'].value_counts().sort_index().reset_index()
cluster_sizes.columns = ['Cluster', 'Size']
print("Cluster Sizes:")
print(cluster_sizes)

# Display cluster centers (scaled feature means)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_env.columns)
centers['Cluster'] = centers.index
centers = centers[['Cluster'] + list(features_env.columns)]
print("\nCluster Centers (scaled features):")
print(centers)

# Plot cluster centers for visual comparison
plt.figure(figsize=(8, 5))
centers_plot = centers.set_index('Cluster')
centers_plot.plot(kind='bar')
plt.title('Cluster Centers (scaled features)')
plt.xlabel('Cluster')
plt.ylabel('Feature value (scaled)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_env)
snapshot90.loc[:, 'PC1'] = pca_result[:, 0]
snapshot90.loc[:, 'PC2'] = pca_result[:, 1]

plt.figure(figsize=(6, 5))
for cluster in sorted(snapshot90['Cluster'].unique()):
    mask = snapshot90['Cluster'] == cluster
    plt.scatter(snapshot90.loc[mask, 'PC1'], snapshot90.loc[mask, 'PC2'], label=f'Cluster {cluster}', alpha=0.7)
plt.title('PCA of Indicators Colored by Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()


# In[36]:


clustered = snapshot90.groupby('Cluster')['Country Name'].apply(list)

for cluster, countries in clustered.items():
    n = len(countries)
    cols = 4
    rows = (n + cols - 1) // cols 
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = c * rows + r
            row.append(countries[idx] if idx < n else "")
        matrix.append(row)

    print(f"Cluster {cluster} ({n} countries):")
    for row in matrix:
        print("  ".join(f"{item:20}" for item in row))
    print()


# ### Comparison
# 
# | Archetype                          | 1990 Center (≈)                                               | Count ’90 | 2023 Center (≈)                                                | Count ’23 |
# |------------------------------------|--------------------------------------------------------------|-----------|---------------------------------------------------------------|-----------|
# | High-Emitter Industrialists    | CO₂ ↑, Renew ↓, Power ↑, GDP ↑                                | 42        | CO₂ ↑, Renew ↓, Power ↑, GDP ↑                                 | 31        |
# | Renewables-Dependent Low-Income| CO₂ ≈ 0, Renew ↑ ↑, Power ↓, GDP ↓                            | 30        | CO₂ ≈ 0, Renew ↑ ↑, Power ↓, GDP ↓                             | 28        |
# | Mixed Mid-Tier                 | CO₂ ≈ 0, Renew moderate, Power moderate, GDP moderate         | 21        | CO₂ ≈ 0, Renew moderate, Power moderate, GDP moderate          | 49        |
# | Nordic-Style Green Consumers   | *Not so separable in 1990, thus the last cluster is not regarded  as a match*                                      | 0         | CO₂ ≈ 0, Renew ↑, Power ↑, GDP ↑                              | 4         |
# 
# 
# High-Emitter Industrialists:
# 
# - 1990: Cluster 2 (42 countries) – large emitters, heavy power use, high GDP, minimal renewables.
# - 2023: Cluster 3 (31 countries) – still the fossil-fuel giants, but slightly fewer as some have decarbonized or fallen behind.
# 
# Renewables-Dependent Low-Income:
# 
# - 1990: Cluster 3 (30) – very high share of traditional renewables, but minimal modern power use/GDP.
# - 2023: Cluster 2 (28) – these countries still rely heavily on renewables (often biomass/hydro) but haven’t built out wider energy infrastructure.
# 
# Mixed Mid-Tier:
# 
# - 1990: Cluster 0 (21) – a handful of emerging economies with modest footprints on all four axes.
# - 2023: Cluster 0 (49) – this group has more than doubled, as dozens of formerly extreme countries moved into a “balanced” middle, reflecting broader energy access, moderate decarbonization, and rising incomes.
# 
# Nordic-Style Green Consumers:
# 
# - 1990: No clear “green consumer” outlier, not distinguished by renewable metrics.
# - 2023: Cluster 1 (4: Finland, Iceland, Norway, Sweden) emerges distinctly: they combine high renewable shares with high power use and high GDP, and nearly zero CO₂ per capita.
# 
# ### Conclusion
# 
# Extremes have softened.
# 
# - The ultra-industrial cluster shrank (42to 31) as some heavy-emitters have diversified energy mixes or slowed growth.
# 
# - The very low-income renewables cluster stayed roughly the same (30to 28), reflecting persistent infrastructure gaps.
# 
# The “broad middle” expanded.
# 
# - Mixed mid-tier economies more than doubled (21 to 49), showing how many countries have climbed from extremes into moderate, balanced energy and carbon profiles.
# 
# A new “green consumer” group emerged.
# 
# - In 1990, renewables were almost purely traditional (biomass, hydro). By 2023, a small elite (the Nordics) stands out as high-income, high-consumption and high-renewables, signaling advanced decarbonization.
# 
# Furthermore:
# - We can observe a global convergence toward the middle. Many nations have achieved at least mid-level energy access, moderate emissions, and rising GDP, reducing the share of strict extremes.
# - There is a distinct cluster of wealthy, high-renewables countries reveals best-practice models for decoupling consumption from emissions.
# - Thre is an enduring divide at the bottom. Dozens of low-income, renewables-dependent countries still need massive investment in modern power systems.
# 

# ## Regression modeling

# The Regression questions we addressed are:
# 
# Question 1: What economic or social variables best predict lower inflation volatility across nations?
# 
# Question 2: To what extent can internet access explain differences in fertility rates, controlling for female labour-force participation and education?
# 
# Question 3: What factors predict the female labor force participation rate?
# 
# Question 4: Do more open economies (with higher trade and investment relative to GDP) experience faster growth? 
# 

# ### Question 1: What economic or social variables best predict lower inflation volatility across nations?

# In[33]:


import statsmodels.api as sm

df_reg1 = df_filtered_1990_2023[['Inflation', 'GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']].dropna()
df_reg1['Inflation_vol'] = df_reg1['Inflation'].rolling(window=3).std()
df_reg1 = df_reg1.dropna()

X = df_reg1[['GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']]
y = df_reg1['Inflation_vol']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# **Key Takeaways from the OLS Regression**
# 
# | Metric                        | Value                 | Interpretation                                                                                                                                                                  |
# | ----------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **R²**                        | 0.006                 | The model explains only **0.6%** of the variance in inflation volatility. This is **very weak**, indicating that these predictors collectively do not explain the outcome well. |
# | **F-statistic**               | 6.052, *p = 0.000075* | The model as a whole is statistically significant, but only marginally  and the low R² makes it practically weak.                                                              |
# | **Durbin-Watson**             | 0.679                 | Indicates **strong positive autocorrelation** in residuals, a major violation of OLS assumptions.                                                                              |
# | **Condition Number**          | 1.52e+05              | This is very high, suggesting **multicollinearity or numerical instability** in the predictors.                                                                                 |
# | **Skew / Kurtosis / Omnibus** | Severe non-normality  | The residuals are **highly skewed and leptokurtic**, which also violates OLS assumptions.                                                                                       |
# 

# **Interpretation of Coefficients**
# 
# | Variable            | Coefficient | p-value   | Interpretation                                                                                                                                                                                      |
# | ------------------- | ----------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **GDP\_pc**         | +0.00000437 | 0.983     | **Not significant**. Changes in GDP per capita have **no meaningful effect** on inflation volatility.                                                                                               |
# | **GINI**            | **+0.746**  | **0.015** | **Statistically significant**. Higher income inequality is associated with **greater inflation volatility**.                                                                                        |
# | **Internet\_Use**   | **–0.287**  | **0.007** | **Statistically significant**. Higher internet penetration is associated with **lower inflation volatility**, possibly through better financial access, information flow, or economic integration. |
# | **Health\_Exp\_pc** | +0.0012     | 0.563     | **Not significant**. Per capita health spending does **not affect** inflation volatility in this model.                                                                                             |
# 

# **Problems With the Model**
# 
# - Extremely low explanatory power (R² = 0.006).
# 
# - Strong residual autocorrelation (Durbin-Watson < 1.0).
# 
# - Multicollinearity warning (Condition number > 30).
# 
# - Non-normal residuals (Omnibus and Jarque-Bera tests highly significant).
# 
# - Coefficient magnitudes are small, especially GDP_pc and Health_Exp_pc.

# In order to dive dipper or rather to investigate this problem we made use of the `LASSO model`.

# **LASSO MODEL**

# In[34]:


# LASSO Regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# calculate inflation volatility using a rolling window, and drop rows with NaNs after rolling
df = df_filtered_1990_2023[['Inflation', 'GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']].dropna()
df['Inflation_vol'] = df['Inflation'].rolling(window=3).std()
df = df.dropna()

# Define predictors and target
X = df[['GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']]
y = df['Inflation_vol']

# Standardize the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit LASSO with cross-validation
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Output results
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
lasso_score = lasso.score(X_test, y_test)
lasso_alpha = lasso.alpha_

# Print summary
print("LASSO Coefficients:\n", lasso_coef)
print("\nTest R² Score:", round(lasso_score, 4))
print("Optimal Alpha:", round(lasso_alpha, 6))


# **Key Interpretations**
# 
# - Internet_Use (strongest effect): Suggests digital inclusion reduces macroeconomic instability. Possibly by improving access to market info, digital finance, and reducing informal cash-based shocks.
# 
# - GINI: High inequality destabilizes prices, consistent with prior research.
# 
# - Health_Exp_pc: Retained but small, possibly noise or related to how governments allocate fiscal resources during volatility.
# 
# - GDP_pc was eliminated, indicating national income by itself does not consistently explain inflation volatility.
# 
# - R² ≈ 0.006, the model has almost no predictive value in practice. It identifies associations, not strong explanations.

# In[35]:


pip install linearmodels


# **Fixed Effects model**

# In[36]:


import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from statsmodels.tools.tools import add_constant

# Ensure panel is structure: MultiIndex (Country, Year)
df = df_filtered_1990_2023.copy()
df = df[['Country Code', 'Year', 'Inflation', 'GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']].dropna()

# Calculate inflation volatility (3-year rolling std) per country
df['Inflation_vol'] = df.groupby('Country Code')['Inflation'].transform(lambda x: x.rolling(3).std())
df = df.dropna()

# Set MultiIndex
df = df.set_index(['Country Code', 'Year'])

# Define dependent and independent variables
y = df['Inflation_vol']
X = df[['GDP_pc', 'GINI', 'Internet_Use', 'Health_Exp_pc']]

# Fit fixed effects model (within estimator)
model = PanelOLS(y, X, entity_effects=True)
results = model.fit()

# Print summary
print(results.summary)


# **Key Result Summary**
# 
# | Metric                     | Value               | Interpretation                                                                                                               |
# | -------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
# | **R² (Within)**            | 0.0087              | The model explains only **0.87%** of variation in inflation volatility **within countries over time**. Low predictive power. |
# | **F-statistic**            | 7.64 (p < .001)     | Overall model is **statistically significant**, but practically weak due to low R².                                          |
# | **F-test for Poolability** | 2.4008 (p = 0.0000) | Confirms fixed effects are appropriate, **countries are not poolable**.                                                     |
# | **Entities**               | 112 countries       | Strong panel dataset                                                                                                         |
# | **Time periods**           | 32 years            | Sufficiently long for panel analysis                                                                                         |
# 

# **Interpretation of Coefficients**
# 
# | Variable            | Coeff   | p-value | Significance      | Interpretation                                                                    |
# | ------------------- | ------- | ------- | ----------------- | --------------------------------------------------------------------------------- |
# | **GDP\_pc**         | –0.0001 | 0.689   | Not Significant | Economic development **does not predict** changes in inflation volatility.        |
# | **GINI**            | +1.72   | 0.005   | Significant     | As **inequality increases**, **inflation volatility increases** within a country. |
# | **Internet\_Use**   | –0.25   | 0.007   | Significant     | More **internet penetration reduces** inflation volatility over time.             |
# | **Health\_Exp\_pc** | +0.0046 | 0.14    | Not Significant | Weak and insignificant; may not meaningfully relate to inflation stability.       |
# 

# **Model Comparison and Justification**

# **1. Ordinary Least Squares (OLS)**
# 
# OLS regression offered an initial exploratory view of relationships between inflation volatility and predictors such as GDP per capita, GINI index, internet usage, and health expenditure. While the model produced statistically significant coefficients for GINI and Internet_Use, its overall explanatory power was extremely low (R² ≈ 0.006). Furthermore, diagnostics revealed:
# 
# High multicollinearity (Condition Number ≈ 1.5e+05),
# 
# Strong residual autocorrelation (Durbin-Watson < 1),
# 
# Violation of normality assumptions.
# 
# **Conclusion:** OLS results were unreliable due to multiple assumption violations and very weak explanatory power.
# 
# **2. LASSO Regression**
# 
# LASSO regression improved on OLS by:
# 
# Automatically performing feature selection,
# 
# Reducing the risk of overfitting through regularization.
# 
# It retained GINI and Internet_Use as relevant predictors of inflation volatility, aligning with OLS findings. However, the model’s predictive power remained weak (R² ≈ 0.0058) and, by design, LASSO focuses on prediction, not causal inference.
# 
# **Conclusion:** LASSO confirmed predictor relevance but lacked policy interpretability and explanatory power.
# 
# **3. Fixed Effects Panel Regression**
# 
# The Fixed Effects (FE) model addresses the shortcomings of both OLS and LASSO by:
# 
# Controlling for unobserved, time-invariant country-specific factors,
# 
# Estimating effects based only on within-country changes over time.
# 
# **Key findings:**
# 
# GINI had a positive and significant effect on inflation volatility (p = 0.005),
# 
# Internet_Use had a negative and significant effect (p = 0.007),
# 
# GDP_pc and Health_Exp_pc had no significant effects.
# 
# Though the R² (within) was low (0.0087), this model offered the most credible interpretation of relationships over time, controlling for all country-specific omitted variables.
# 
# **Conclusion:** The FE model is statistically valid and conceptually aligned with the research objective, making it the most appropriate choice for identifying social and economic factors associated with inflation stability.
# 
# **Final Recommendation**
# 
# We selected the Fixed Effects model as the best method to answer the question. Despite modest explanatory power, it provided the most trustworthy insight into the dynamic relationship between inequality, digital access, and inflation volatility. The consistent significance of Internet access and income inequality across all models further reinforces their relevance as policy levers for macroeconomic stability.

# ### Question 2:  To what extent can internet access explain differences in fertility rates, controlling for female labour-force participation and education?

# In[37]:


df_reg2 = df_filtered_1990_2023[['Fertility_Rate', 'Internet_Use', 'Female_LFP', 'Tertiary_Enroll']].dropna()
X = sm.add_constant(df_reg2[['Internet_Use', 'Female_LFP', 'Tertiary_Enroll']])
y = df_reg2['Fertility_Rate']
model2 = sm.OLS(y, X).fit()
print(model2.summary())


# **Key Result Summary**
# | Metric            | Value               | Interpretation                                                                           |
# | ----------------- | ------------------- | ---------------------------------------------------------------------------------------- |
# | **R²**            | 0.519               | The model explains 52% of the variation in fertility rates, a strong model fit |
# | **F-statistic**   | 1371.0 (p < 0.0001) | The overall model is highly statistically significant                              |
# | **No. of Obs**    | 3,808               | Large sample; reliable estimates                                                       |
# | **Durbin-Watson** | 0.083               | Severe positive autocorrelation in residuals; violates OLS             |
# | **Cond. Number**  | 277                 | Acceptable; no serious multicollinearity                                               |
# 

# **Interpretation of Coefficients**
# | Variable             | Coeff   | p-value | Effect on Fertility | Interpretation                                                                                                                                                      |
# | -------------------- | ------- | ------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use**    | –0.0041 | 0.000   | **Negative**        | 1% increase in internet use is associated with a **0.0041 unit decrease** in fertility rate, holding other variables constant. This is statistically significant. |
# | **Female\_LFP**      | +0.0199 | 0.000   | **Positive**        | 1% increase in female labor participation is associated with a **0.0199 unit increase** in fertility. Surprisingly **positive**, and significant.                 |
# | **Tertiary\_Enroll** | –0.0371 | 0.000   | **Negative**        | 1% increase in tertiary enrollment leads to a **0.0371 unit drop** in fertility rate. Strongest effect among predictors.                                          |
# 

# **Problem with the model**
# | Issue                                     | Implication                                                                                                                                       |
# | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Durbin-Watson = 0.083**                 | Indicates **strong residual autocorrelation**. This violates OLS assumptions and **inflates test statistics**, making the model overly confident. |
# | **Residual normality tests (Omnibus/JB)** | Deviations from perfect normality, though not extreme.                                                                                            |
# 

# **LASSO Regression Code**

# In[38]:


import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare the dataset
df = df_filtered_1990_2023[['Fertility_Rate', 'Internet_Use', 'Female_LFP', 'Tertiary_Enroll']].dropna()

# Define X and y
X = df[['Internet_Use', 'Female_LFP', 'Tertiary_Enroll']]
y = df['Fertility_Rate']

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit LASSO with cross-validation
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Output results
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
lasso_score = lasso.score(X_test, y_test)
lasso_alpha = lasso.alpha_

# Display results
print("LASSO Coefficients:\n", lasso_coef)
print("\n Test R² Score:", round(lasso_score, 4))
print("Optimal Alpha (Penalty Strength):", round(lasso_alpha, 6))


# **Key Result Summary**
# | Metric            | Value    | Interpretation                                                                                                                                 |
# | ----------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Test R² Score** | 0.5255   | The model explains about **52.6% of the variation** in fertility rates on unseen data, a **strong predictive performance**.                   |
# | **Optimal Alpha** | 0.001125 | This is the penalty strength that balances bias and variance. The value is small, so most predictors were retained rather than shrunk to zero. |
# 

# **Interpretation of Coefficients**
# 
# | Variable             | Coefficient | Direction   | Interpretation                                                                   |
# | -------------------- | ----------- | ----------- | -------------------------------------------------------------------------------- |
# | **Internet\_Use**    | –0.1474     | Negative | As internet use increases, fertility rate **decreases**.                         |
# | **Female\_LFP**      | +0.2860     | Positive | Higher female labor force participation is associated with **higher** fertility. |
# | **Tertiary\_Enroll** | –1.0334     | Negative | Higher education levels correlate strongly with **lower** fertility rates.       |
# 

# Since the LASSO result same as the OSL result,  we diver deep to Fixed Effects Panel Regression to control for country effects

# **Fixed Effects Panel Regression**

# In[40]:


import pandas as pd
from linearmodels.panel import PanelOLS

# Select and drop missing values
df = df_filtered_1990_2023[['Country Code', 'Year', 'Fertility_Rate', 'Internet_Use', 'Female_LFP', 'Tertiary_Enroll']].dropna()

# Set panel structure (MultiIndex by country and year)
df = df.set_index(['Country Code', 'Year'])

# Define dependent and independent variables
y = df['Fertility_Rate']
X = df[['Internet_Use', 'Female_LFP', 'Tertiary_Enroll']]

# Fit Fixed Effects model (country and year fixed effects)
model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit()

# Output the results
print(results.summary)


# **Key Result Summary**
# | Metric                   | Value              | Interpretation                                                                                                       |
# | ------------------------ | ------------------ | --------------------------------------------------------------------------- |                                 
# | **R² (Within)**          | **-1.2090**        | **Problematic**: Indicates poor model fit within countries over time (potential overfitting or multicollinearity) |
# | **R² (Overall)**         | 0.0099             | Model explains \~1% of total variation                                                                               |
# | **F-statistic**          | 429.15 (p = 0.000) | Model is statistically significant                                                                                   |
# | **Poolability Test**     | p < 0.000          | Confirms the need for **fixed effects** (countries and years **cannot be pooled**)                                   |
# 

# **Coefficient Interpretation**
# | Variable             | Coefficient | p-value | Interpretation                                                                                                                                                                                                                           |
# | -------------------- | ----------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use**    | **+0.0117** | <0.001  | A 1% increase in internet access is associated with a **0.0117 unit increase** in fertility rate, **within countries over time**. This is **statistically significant**, but **surprising**, as previous models found a negative effect. |
# | **Female\_LFP**      | **–0.0057** | <0.001  | As female labor force participation increases by 1%, fertility decreases slightly (–0.0057 units). This is **statistically significant** and aligns with expectations.                                                                   |
# | **Tertiary\_Enroll** | **+0.0060** | <0.001  | A 1% increase in higher education enrollment is associated with a **slight increase** in fertility. This **contradicts** earlier findings from OLS and LASSO.                                                                            |
# 

# Since the `R²` result poorly fit Fixed Effects Panel Regression model, we try to find out the issue by checking the `Multicollinearity (Variance Inflation Factors) and Interaction Terms`.

# **Variance Inflation Factor (VIF)**

# In[42]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Drop missing values and select predictors
df_vif = df_filtered_1990_2023[['Internet_Use', 'Female_LFP', 'Tertiary_Enroll']].dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_vif)

# Create DataFrame
X_df = pd.DataFrame(X_scaled, columns=['Internet_Use', 'Female_LFP', 'Tertiary_Enroll'])

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Variable"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print(vif_data)


# **Variance Inflation Factor (VIF) Interpretation**
# | Variable             | VIF  | Interpretation                                                        |
# | -------------------- | ---- | --------------------------------------------------------------------- |
# | **Internet\_Use**    | 2.21 | Low-to-moderate correlation with other predictors. |
# | **Female\_LFP**      | 1.01 | **Very low** multicollinearity, ideal.                               |
# | **Tertiary\_Enroll** | 2.19 | Low-to-moderate correlation. **Acceptable.**                          |
# 

# **Interaction Terms**

# In[43]:


from linearmodels.panel import PanelOLS

# Prepare dataset with interactions
df = df_filtered_1990_2023[['Country Code', 'Year', 'Fertility_Rate', 'Internet_Use', 'Female_LFP', 'Tertiary_Enroll']].dropna()

# Interaction terms
df['Internet_Edu'] = df['Internet_Use'] * df['Tertiary_Enroll']
df['Internet_LFP'] = df['Internet_Use'] * df['Female_LFP']

# Set MultiIndex for panel structure
df = df.set_index(['Country Code', 'Year'])

# Define dependent and independent variables (with interactions)
y = df['Fertility_Rate']
X = df[['Internet_Use', 'Female_LFP', 'Tertiary_Enroll', 'Internet_Edu', 'Internet_LFP']]

# Fit Fixed Effects model (entity and time effects)
model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit()

# Display output
print(results.summary)


# **Coefficient Interpretation**
# | Variable             | Coeff   | p-value | Effect on Fertility                | Interpretation                                                                                                                  |
# | -------------------- | ------- | ------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use**    | –0.0034 | 0.0001  | **Significant** negative effect | As internet use increases, fertility declines, consistent with original hypothesis.                                            |
# | **Female\_LFP**      | –0.0087 | 0.0000  | Negative effect                 | Female labor participation is now **consistently associated with lower fertility**.                                             |
# | **Tertiary\_Enroll** | –0.0019 | 0.0116  | Negative effect                 | Higher education continues to reduce fertility, though effect is small.                                                         |
# | **Internet\_Edu**    | +0.0001 | 0.0000  | Positive interaction            | Internet **dampens** the negative effect of education. Educated women with internet access may be more likely to have children. |
# | **Internet\_LFP**    | +0.0001 | 0.0000  | Positive interaction            | Internet also **moderates** the fertility-reducing effect of working, possibly via work–life balance or remote work.           |
# 

# **Negative R² (within)**: After the checking the Multicollinearity (Variance Inflation Factors) and Interaction Terms `R²` still remain negative.

# **Model Comparison**
# 
# | Model                            | Key Findings                                                                           | Strengths                                         | Weaknesses                                   |
# | -------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------- |
# | **OLS Regression**               | Internet access significantly **reduces fertility**; model R² = 0.52                   | Simple, interpretable                             | Biased by unobserved country-level factors   |
# | **LASSO Regression**             | Internet, LFP, and Education retained; strong predictor set (R² = 0.53)                | Handles multicollinearity, selects key predictors | Predictive only; not causal                  |
# | **Fixed Effects Panel**   | Internet effect reversed (positive); R² (within) negative                              | Controls for country & year                       | Poor fit, possible overfitting               |
# | **VIF Diagnostic**               | All variables had **low multicollinearity** (VIF < 2.5)                                | Confirms model stability                          | Diagnostic tool only                         |
# | **Fixed Effects + Interactions** | Internet directly reduces fertility; also **moderates** the effects of LFP & education | Most robust model; shows nuanced dynamics         | R² (within) still negative but interpretable |
# 

# **Conclusion and  Model Recommendation**
# 
# The Fixed Effects Panel Model with Interaction Terms best answers the question. It controls for hidden country and time factors, offers detailed insights into both direct and moderating effects, and provides the most policy-relevant evidence for how digital inclusion influences demographic behavior.
# 
# 

# ### Question 3: What factors predict the female labor force participation rate?

# In[37]:


df_reg3 = df_filtered_1990_2023[['Female_LFP', 'Internet_Use', 'Health_Exp_pc', 'Tertiary_Enroll', 'GINI']].dropna()
X = sm.add_constant(df_reg3.drop(columns='Female_LFP'))
y = df_reg3['Female_LFP']
model3 = sm.OLS(y, X).fit()
print(model3.summary())


# **Key Result Summary**
# | Metric               | Value    | Interpretation                                                                                                                                                        |
# | -------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **R²**            | 0.045    | The model explains **4.5%** of the variation in female labor force participation. This is **statistically significant** (p < 0.001). |
# | **F-statistic**      | 45.24    | The model is globally significant (p < 0.0001).                                                                                                                       |
# | **Observations**     | 3,808    | Large dataset reliable statistical power.                                                                                                                           |
# | **Durbin-Watson**    | 0.072    | Very low **strong positive autocorrelation** in residuals. Violates independence assumption.                                                                        |
# | **Condition Number** | 1.18e+04 | High **potential multicollinearity or scale imbalance** between predictors.                                                                                         |
# 

# **Coefficient Interpretation**
# | Variable              | Coefficient | p-value | Interpretation                                                                                                                                                                                                                         |
# | --------------------- | ----------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use**     | **+0.0276** | 0.013   | 1% increase in internet access is associated with a **0.0276 percentage point increase** in female labor force participation, holding other factors constant. **Positive and significant.**                                          |
# | **Health\_Exp\_pc**   | **+0.0020** | <0.001  | Higher health spending is associated with greater female LFP. This may signal better healthcare access and support services enabling women to work.                                                                                    |
# | **Tertiary\_Enroll**  | **–0.0636** | <0.001  | Surprisingly, higher female tertiary enrollment correlates with **lower female LFP**, likely because **more women are in school instead of the workforce**. This is **plausible**, especially in developing contexts.                 |
# | **GINI (Inequality)** | **+0.1376** | <0.001  | Higher inequality is **associated with higher female labor participation**. This is counterintuitive and may reflect that in **more unequal economies**, women are compelled to work out of economic necessity. |
# 

# **Problem with the OSL Model**
# | Issue                     | Meaning                                                                                                               |
# | ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
# | **Low R² (0.045)**        | The model does not capture most of the variation in female LFP; suggests many relevant factors are missing.          |
# | **Durbin-Watson = 0.072** | **Severe autocorrelation**, model residuals are not independent across time.                                         |
# | **High Condition Number** | Potential **multicollinearity or scale imbalances**, especially if GINI and health spending are on different scales. |
# 

# **Fixed Effects Panel Regression** 

# In[51]:


import pandas as pd
from linearmodels.panel import PanelOLS

# Select relevant columns and drop missing values
df_fe = df_filtered_1990_2023[['Country Code', 'Year', 'Female_LFP', 'Internet_Use', 'Health_Exp_pc', 'Tertiary_Enroll', 'GINI']].dropna()

# Set panel structure (MultiIndex for country and year)
df_fe = df_fe.set_index(['Country Code', 'Year'])

# Define outcome and predictors
y = df_fe['Female_LFP']
X = df_fe[['Internet_Use', 'Health_Exp_pc', 'Tertiary_Enroll', 'GINI']]

# Fit Fixed Effects Model (both country and time effects)
model_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
results_fe = model_fe.fit()

# Display results
print(results_fe.summary)


# **Key Result Summary**
# | Metric                                | Value                                                                                                     | Interpretation                                                                                                               |
# | ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
# | **R² (Within)**                       | **0.1482**                                                                                                | The model explains **14.8% of the variation** in female LFP **within countries over time**, an improvement over OLS (4.5%). |
# | **R² (Between)**                      | 0.0708                                                                                                    | Explains \~7% of variation **between countries**.                                                                            |
# | **R² (Overall)**                      | 0.0712                                                                                                    | Total explanatory power is modest, but this is expected in social science panel data.                                       |
# | **F-statistic**            |  (p < .001) |  Statistically significant overall. The included predictors explain a non-random portion of the variation.                                                                                                                            |
# | **F-test for Poolability** |      (p < .001)     | Fixed effects are appropriate, confirms that countries and years cannot be pooled without bias.                                                                                                                             |
# 

# **Coefficient Interpretation**
# | Variable              | Coefficient | p-value | Direction   | Interpretation                                                                                                                                                      |
# | --------------------- | ----------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Internet\_Use**     | **+0.0690** | <0.001  | Positive | A 1% increase in internet access is associated with a **0.069 percentage point increase** in female LFP, holding other factors constant.                            |
# | **Health\_Exp\_pc**   | **+0.0006** | <0.001  | Positive | Higher health spending slightly boosts female LFP, likely by improving maternal health, access to care, and reducing childcare burdens.                             |
# | **Tertiary\_Enroll**  | **+0.0336** | <0.001  | Positive | Contrary to earlier OLS results, here **higher education correlates with more women working**, a theoretically sound and now fixed-effect-controlled relationship. |
# | **GINI** (inequality) | **–0.0471** | 0.030   | Negative | More unequal countries experience **lower female labor force participation**, likely due to structural exclusion and reduced economic opportunity for women.        |
# 

# **Conclusion**
# 
# The **Fixed Effects model** confirms that internet access, education, and public health investment are key enablers of female labor force participation, while income inequality suppresses it. These effects are statistically significant, directionally plausible, and robust after controlling for both country-specific traits and global year trends.

# **Model Comparison**

# 
# 
# **OLS Regression Summary**
# | Feature                  | Value                                                                                                     |
# | ------------------------ | --------------------------------------------------------------------------------------------------------- |
# | **R²**                   | 0.045 (4.5%)                                                                                              |
# | **Top Predictors**       | All predictors significant                                                                                |
# | **Direction of Effects** | Mixed: Education showed **negative**, Internet & Health **positive**, GINI **positive**                   |
# | **Main Limitation**      | Does **not control for unobserved differences** across countries or years; prone to omitted variable bias |
# | **Conclusion**           | Suggests weak but significant relationships, but possibly biased or misleading                            |
# 

# **Key OLS Insight**
# 
# Internet access and health spending correlate with increased female LFP, but tertiary education showed a counterintuitive negative effect, and inequality appeared to increase participation, likely due to unmeasured national differences.

# **Fixed Effects Panel Regression Summary**
# 
# | Feature                  | Value                                                                                 |
# | ------------------------ | ------------------------------------------------------------------------------------- |
# | **R² (Within)**          | 0.148 (14.8%)                                                                         |
# | **Controls**             | Country & Year Fixed Effects                                                          |
# | **Direction of Effects** | All effects **theoretically consistent**: Internet, Education, Health, GINI |
# | **Main Advantage**       | Removes **time-invariant country-specific bias** and **global time shocks**           |
# | **Conclusion**           | More reliable and interpretable estimates of within-country effects over time         |
# 

# **Key Fixed Effects Insight**
# 
# When accounting for national and temporal differences:
# 
# - Internet access has a strong, positive effect on female LFP
# 
# - Education now shows a positive effect (correcting the OLS bias)
# 
# - Inequality reduces female employment, and
# 
# - Health expenditure remains a positive, supportive factor

# **Conclusion and Recommendation**
# 
# We use Fixed Effects Panel Model as the primary method for policy conclusions. It yields more valid, time-aware, and country-controlled insights. OLS results are useful for exploratory comparison but suffer from omitted variable bias.

# ### Question 4: Do more open economies (with higher trade and investment relative to GDP) experience faster growth?

# In[38]:


df_reg4 = df_filtered_1990_2023[['GDP_growth', 'FDI', 'Exports', 'Imports']].dropna()
df_reg4['Trade_Openness'] = df_reg4['Exports'] + df_reg4['Imports']
X = sm.add_constant(df_reg4[['Trade_Openness', 'FDI']])
y = df_reg4['GDP_growth']
model4 = sm.OLS(y, X).fit()
print(model4.summary())


# **Key Result Summary**
# 
# | Metric                  | Value           | Interpretation                                                                                     |
# | ----------------------- | --------------- | -------------------------------------------------------------------------------------------------- |
# | **R²**                  | 0.004           | The model explains **only 0.4%** of the variation in GDP growth, **very weak explanatory power**. |
# | **F-statistic**         | 8.16 (p < .001) | The model is **statistically significant**, but practically weak.                                  |
# | **No. of Observations** | 3,808           | Large sample, strong statistical power.                                                            |
# | **Durbin-Watson**       | 1.492           | Mild autocorrelation in residuals, not severe, but slightly below ideal (≈ 2).                    |
# | **Jarque-Bera**         | Very large      | Indicates **non-normal residuals**, GDP growth is likely skewed or has outliers.                  |
# 

# **Coefficient Interpretation**
# | Predictor           | Coefficient | p-value | Direction   | Interpretation                                                                                                                                                                                       |
# | ------------------- | ----------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Trade\_Openness** | **+0.0033** | 0.021   | Positive | 1-unit increase in trade openness (exports + imports as % of GDP) is associated with a **0.0033 percentage point increase** in GDP growth. Effect is statistically significant but **very small**. |
# | **FDI**             | **+0.0092** | 0.006   | Positive | 1-unit increase in FDI (% of GDP) is linked to a **0.0092 percentage point increase** in GDP growth, small, but statistically significant.                                                        |
# | **Constant**        | 3.0413      | <0.001  | -+           | Baseline GDP growth when Trade\_Openness and FDI are zero, **mostly theoretical**.                                                                                                                  |
# 

# **Conclusion**
# 
# The results suggest that more open economies tend to grow slightly faster, but the impact of trade and investment on growth is statistically significant yet economically marginal. The model’s explanatory power is extremely limited, indicating the need to include broader structural, institutional, and macroeconomic variables to better understand what drives economic growth.

# **Fixed Effects Panel Regression**

# In[55]:


import pandas as pd
from linearmodels.panel import PanelOLS

# Prepare and clean data
df_panel = df_filtered_1990_2023[['Country Code', 'Year', 'GDP_growth', 'FDI', 'Exports', 'Imports']].dropna()

# Create Trade Openness variable
df_panel['Trade_Openness'] = df_panel['Exports'] + df_panel['Imports']

# Set MultiIndex for panel structure
df_panel = df_panel.set_index(['Country Code', 'Year'])

# Define target and predictors
y = df_panel['GDP_growth']
X = df_panel[['Trade_Openness', 'FDI']]

# Fit fixed effects model (with entity and time effects)
model_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
results_fe = model_fe.fit()

# Display results
print(results_fe.summary)


# **Key Result Summary**
# 
# | Metric                     | Value            | Interpretation                                                                                                           |
# | -------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
# | **R² (Within)**            | **0.0042**       | Only **0.42%** of variation in GDP growth **within countries over time** is explained, extremely low explanatory power. |
# | **R² (Between)**           | 0.2303           | Trade and FDI explain **23% of growth differences between countries**, but not within them.                              |
# | **F-statistic (overall)**  | 1.94 (p = 0.144) | Model is **not statistically significant**.                                                                              |
# | **F-test for Poolability** | p < 0.001        | Confirms that **fixed effects are necessary** (countries are not poolable).                                              |
# 

# **Coefficient Interpretation**
# 
# | Predictor           | Coeff   | p-value | Direction                        | Interpretation                                                                                                                                              |
# | ------------------- | ------- | ------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Trade\_Openness** | +0.0060 | 0.148   | Positive, but not significant | A 1-unit increase in trade openness (% GDP) is associated with a **0.006 percentage point increase** in growth, **small and statistically insignificant**. |
# | **FDI**             | +0.0044 | 0.187   | Positive, but not significant | A 1-unit increase in FDI (% of GDP) corresponds to a **0.0044 percentage point growth increase**, also **not statistically significant**.                  |
# 

# **Conclusion**
# 
# The fixed effects panel model finds no statistically significant relationship between trade openness or FDI and GDP growth within countries over time. While more open countries may grow more on average (between-country effect), increasing trade or investment within a country does not guarantee faster growth in the short to medium term.

# For this reason we try to enhance the model to test whether openness and investment have delayed `(lagged)` effects on GDP growth and whether their impact depends on institutional or economic context `(interactions)`.

# In[56]:


# Lag predictors within each country
df_lagged = df_filtered_1990_2023[['Country Code', 'Year', 'GDP_growth', 'FDI', 'Exports', 'Imports']].dropna()

# Create Trade_Openness
df_lagged['Trade_Openness'] = df_lagged['Exports'] + df_lagged['Imports']

# Sort and lag
df_lagged.sort_values(['Country Code', 'Year'], inplace=True)
df_lagged['Trade_Openness_L1'] = df_lagged.groupby('Country Code')['Trade_Openness'].shift(1)
df_lagged['FDI_L1'] = df_lagged.groupby('Country Code')['FDI'].shift(1)

# Drop NA (due to lag)
df_lagged = df_lagged.dropna()

# Set panel structure
df_lagged = df_lagged.set_index(['Country Code', 'Year'])

# Define target and predictors
y = df_lagged['GDP_growth']
X = df_lagged[['Trade_Openness_L1', 'FDI_L1']]

# Fit fixed effects model
from linearmodels.panel import PanelOLS
model_lagged = PanelOLS(y, X, entity_effects=True, time_effects=True)
results_lagged = model_lagged.fit()
print(results_lagged.summary)


# **Key Rsult Summary**
# 
# | Metric                       | Value                                                                          | Interpretation                                                                                                                                |
# | ---------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
# | **R² (Within)**              | **0.0011**                                                                     | The model explains **0.11%** of variation in GDP growth **within countries over time**; **still very weak**, but comparable to prior models. |
# | **R² (Between)**             | 0.3898                                                                         | Stronger association **between countries**; openness is more helpful in distinguishing growth levels across countries.                       |
# | **R² (Overall)**             | 0.1542                                                                         | About 15% of overall variation explained; mostly from between-country differences.                                                           |
# | **F-statistic (p = 0.0165)** | Model is **statistically significant**, though with limited explanatory power. |                                                                                                                                               |
# 

# **Coefficient Interpretation**
# 
# | Variable                | Coefficient | p-value | Interpretation                                                                                                                                                                                                                        |
# | ----------------------- | ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Trade\_Openness\_L1** | **+0.0120** | 0.0043  | A 1% increase in trade openness **last year** is associated with a **0.012 percentage point increase in GDP growth this year**. The effect is **statistically significant** and **slightly stronger than in contemporaneous models**. |
# | **FDI\_L1**             | –0.0009     | 0.7901  | **No significant effect** of lagged FDI on growth; possibly due to noise, delayed absorption, or inefficiency of FDI in certain countries.                                                                                           |
# 

# **Conclusion**
# 
# Lagged trade openness has a small but `statistically significant effect` on economic growth, suggesting that the benefits of global integration are not immediate. However, FDI; even with a lag does not appear to predict growth, and overall the model still explains very little of the within-country growth dynamics.

# In[57]:


# Interaction term: lagged openness × lagged FDI
df_lagged['Open_FDI_L1'] = df_lagged['Trade_Openness_L1'] * df_lagged['FDI_L1']

# Update X to include interaction
X_interact = df_lagged[['Trade_Openness_L1', 'FDI_L1', 'Open_FDI_L1']]

# Fit model
model_interact = PanelOLS(y, X_interact, entity_effects=True, time_effects=True)
results_interact = model_interact.fit()
print(results_interact.summary)


# **Key Result Summary**
# 
# | Metric           | Value             | Interpretation                                                                                                        |
# | ---------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------- |
# | **R² (Within)**  | **0.0012**        | Only **0.12%** of GDP growth variation **within countries over time** is explained; **very weak explanatory power**. |
# | **R² (Between)** | 0.3746            | Trade and FDI explain **37% of differences between countries**.                                                       |
# | **R² (Overall)** | 0.1483            | Total model explains **14.8% of all variation**, mostly across countries.                                             |
# | **F-statistic**  | 3.09 (p = 0.0262) | The model is **statistically significant**, but weak in explanatory power.

# **Coefficient Interpretation**
# 
# | Variable                | Coefficient | p-value    | Interpretation                                                                                                                                                  |
# | ----------------------- | ----------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Trade\_Openness\_L1** | **+0.0116** | **0.0058** | A 1% increase in trade openness **last year** is associated with a **0.0116% increase in GDP growth this year**, **significant** and consistent across models. |
# | **FDI\_L1**             | –0.0090     | 0.2958     | Lagged FDI remains **statistically insignificant**, may not translate into growth within the time window.                                                      |
# | **Open\_FDI\_L1**       | +0.000038   | 0.3063     | The **interaction term is not significant**, no clear evidence that the effect of FDI depends on trade openness.                                               |
# 

# **Conclusion**
# 
# The results indicate that increased trade openness has a small but consistently positive lagged effect on GDP growth, while foreign direct investment (FDI) shows no significant impact, either directly or through interaction with openness. However, the model’s overall explanatory power is low, suggesting that growth dynamics are multifactorial and not strongly driven by openness or FDI alone.

# **Model Comparison Table**
# 
# | **Model**                                | **Key Findings**                                                                                          | **Strengths**                                           | **Weaknesses**                                                               |
# | ---------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------- |
# | **OLS Regression**                       | Trade openness and FDI both had **positive and significant** effects on growth                            | Simple and easy to interpret; all variables significant | Does **not control for country or time-specific effects**; **low R² (0.4%)** |
# | **Fixed Effects**      | Both trade openness and FDI became **insignificant** when controlling for unobserved country/time effects | Controls for **country and year heterogeneity**         | Very low within R² (0.1%); **FDI effect vanished**                           |
# | **Fixed Effects with 1-Year Lag**        | **Lagged trade openness becomes significant**; FDI remains non-significant                                | Captures **delayed effects** of policy                  | Still weak explanatory power (R² = 0.1%); FDI shows no impact                |
# | **Fixed Effects with Lag + Interaction** | Lagged trade openness is significant; interaction term and FDI are **not significant**                    | Tests **synergy** between openness and FDI              | No interaction effect; **R² (within) still very low**                        |
# 

# **Conclusion**
# 
# The initial OLS model showed significant positive effects for both openness and FDI, but after controlling for country- and year-specific unobserved factors using fixed effects models, only **lagged trade openness remained a statistically significant predictor of growth. Its effect, while small, was consistently positive.**
# 
# In contrast, FDI; whether current, lagged, or interacted with openness - was not a significant driver of economic growth, indicating that the quantity of investment alone is insufficient. Other factors such as the quality of institutions, sectoral targeting of FDI, or absorptive capacity likely matter more.
# 
# Despite statistical significance in some models, all regressions exhibited very low explanatory power when looking at within-country variation over time, reinforcing that growth is driven by a broader set of structural and policy variables not captured in these simple models.

# **Policy Implication**
# 
# Trade openness appears to support long-term growth, but FDI requires complementary conditions; such as strong governance, labor skill development, and infrastructure to translate into sustainable economic gains.

# ## Classification

# The classification questions addressed in this notebook are:
# 
# 1. 'High Inequality Prediction': Which combination of development indicators can classify a country as having high inequality, defined by a GINI index above a certain threshold?
# 
# 2. 'Economic Instability Risk Detection': Can we identify countries at risk of economic instability using early warning signals such as rising inflation, declining exports, and infant mortality rates?
# 
# 3. 'High-Income Classification': Using available development indicators, can we classify whether a country is high-income or not? What are the strongest predictors for high-income classification?
# 
# 4. 'CO₂ Emission Classification': Can development indicators help in classifying countries as high versus low CO₂ emitters?
# 
# After question 4, additional parts can be found like 'AUC-ROC Curves', 'Decision Tree Visualization', and 'Hyperparameter Tuning with GridSearchCV' to help visiualize and interpret the data and the outputs.
# 
# The goal with this part is to apply and evaluate multiple classification models including logistic regression, decision trees, and random forests. The emphasis is on the model evaluation, the interpretability, and to draw insights from classification boundaries and feature importance.
# 

# ### Setup and data preparation

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# In[36]:


data = pd.read_csv('clean_world_dev_data.csv')

data = data.dropna(subset=[
    'GINI',
    'Inflation',
    'Exports',
    'CO2_Emissions'
])

data.head()


# Feature selection:

# In[40]:


drop_cols = [col for col in ['GINI'] if col in data.columns]
numeric_features = data.select_dtypes(include=['float64', 'int64']).drop(columns=drop_cols).columns.tolist()


# In[41]:


data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())


# Function to train and evaluate models:

# In[43]:


def train_and_evaluate(X, y, question_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"--- {question_name} - {name} ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        if name == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
            plt.figure(figsize=(8, 5))
            sns.barplot(x=importances, y=importances.index)
            plt.title(f"Top 10 Feature Importances - {question_name}")
            plt.show()


# ### Question 1: Classify countries with high inequality (GINI > 40)

# Threshold for high inequality:

# In[45]:


data_q1 = data.dropna(subset=['GINI'])

data_q1['High_Inequality'] = (data_q1['GINI'] > 40).astype(int)

X_q1 = data_q1[['GDP_pc', 'Inflation', 'Health_Exp_pc', 'Internet_Use', 'Unemployment', 'Mobile_Use', 'Power_Use']]
y_q1 = data_q1['High_Inequality']


# Train-test split:

# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X_q1, y_q1, test_size=0.2, random_state=42)


# Random Forest Classifier:

# In[47]:


rf_q1 = RandomForestClassifier(random_state=42)
rf_q1.fit(X_train, y_train)
y_pred = rf_q1.predict(X_test)


# Evaluation:

# In[48]:


print("Classification Report - High Inequality")
print(classification_report(y_test, y_pred))


# Interpretation
# 
# - The random forest classification model is trying to learn which of the development indicators' combinations are the best in predicting whether a country has a high income inequality (GINI > 40). 
# - The following features are used here: 'GDP_pc', 'Inflaition', 'Unemployment', 'HEalth_Exp_pc', 'Internet_Use', and 'Power_Use'.
# - The model accuracy (83%) and the F1 scores (Macro Average F1: 82%; Weighted Average F1: 83%) are reflecting on how well the features (mentioned above) capture the ineqaulity patterns. 
# - The accuarcy in our case is indicating that most of the time the model is correctly prediciting the observations. Also, the high F1 score means that there is a good balance between precision and recall. So overall, the model is good at predicting both classes, but it is slightly bette at class 0.  
# - For example, if low health expenditure or hihg unemployment are frequent in the countries that have a high GINI, then the model will learn to associate these features with ineqaulity.
# 
# Implications
# 
# - Countries with higher uneplmoyment, lower internet usage, and weaker public services, like health expenditure, may face hihgher ineqaulity,
# - The findings of question 1 maybe good to support policy discussions about investing in social services to reduce inequality.
# - Although, it has to be kept in mind, that correclation does not equal causation. The results of question 1 can be useful for screening, but not for policiy conclusions.

# ### Question 2: Economic instability classification (rising inflation, low exports, low life expectancy)

# Define economic instability based on percentiles:

# In[64]:


data_q2 = data.dropna(subset=['Inflation', 'Exports', 'Life_Expectancy'])

data_q2['Econ_Unstable'] = (
    (data_q2['Inflation'] > data_q2['Inflation'].quantile(0.75)) &
    (data_q2['Exports'] < data_q2['Exports'].quantile(0.25)) &
    (data_q2['Life_Expectancy'] < data_q2['Life_Expectancy'].quantile(0.25))
).astype(int)

X_q2 = data_q2[['GDP_pc', 'Inflation', 'Exports', 'Unemployment', 'Fertility_Rate', 'Health_Exp_pc', 'Power_Use']]
y_q2 = data_q2['Econ_Unstable']

X_train, X_test, y_train, y_test = train_test_split(X_q2, y_q2, test_size=0.2, random_state=42)

logreg_q2 = LogisticRegression(max_iter=1000)
logreg_q2.fit(X_train, y_train)
y_pred = logreg_q2.predict(X_test)

print("Classification Report - Economic Instability")
print(classification_report(y_test, y_pred))


# Interpretation
# 
# - In question 2, the model is using a logistic regression classifier in order to predict economic instability.
# - It is defined as a country with high inflation (top 25%), low exports (bottom 25%), low life expectancy (bottom 25%).
# - These are custom composite proxies that is based on early warning indicators. 
# - The input features are the following: 'GDP_pc', 'Inflation', 'Exports', 'Unemployment', 'Fertility_Rate', 'Health_Exp_pc', and 'Power_Use'.
# - These coefficients in the logistic regression model indicate how each variable contribures to the likelihood of instability.
# - We can see from the output that the model is quiet good at indentifying the stable economies (class 0), but it is having struggels with detecting the unstable economies (class 1). This might be due to class imbalances, that unstable cases are rare compared to stable ones.
# - Further, high fertility and low GDP per capita may increase the likelihood of instability, indicating potential demographic and structureal issues.
# 
# Implications
# 
# - The countries that have those characteristics, mentioned above, should be flagged to be monitored close or to take proactive policy support, for example debt relief or inflation control.
# - An early warning system can help to prioritize the gloval development aids or to signal internal reforms.
# - However, it has to be considered that these results are data-driven but not exhaustive.

# ### Question 3: Classify High-Income countries using GDP_pc approximation

# We will, in the following, define high income as GDP_pc > 12,000 USD.

# In[51]:


data_q3 = data.dropna(subset=['GDP_pc'])

data_q3['High_Income'] = (data_q3['GDP_pc'] > 12000).astype(int)

X_q3 = data_q3[['GDP_pc', 'Health_Exp_pc', 'Internet_Use', 'Fertility_Rate', 'Life_Expectancy', 'Mobile_Use', 'Power_Use']]
y_q3 = data_q3['High_Income']

X_train, X_test, y_train, y_test = train_test_split(X_q3, y_q3, test_size=0.2, random_state=42)

rf_q3 = RandomForestClassifier(random_state=42)
rf_q3.fit(X_train, y_train)
y_pred = rf_q3.predict(X_test)

print("Classification Report - High Income")
print(classification_report(y_test, y_pred))


# Interpretation
# 
# - In question 3, there is an approximation done on income group status by using a hard cutoff. The countries with GDP per capita hihgher than USD 12,0000 are classified as high income. 
# - The random forest classifier uses the following indicators: 'Health_Exp_pc', 'Internet_Use', 'Fertility_Rate','Life_Expectancy', 'Mobile_Use', and 'Power_Use'.
# - The high-income countries tend to show, low fertility rates, hihg internet and mobile usage, hihger health spending, and greater energy use per capita. 
# - It can be said that the features mentioned above are highly predictive due to development patterns.
# - Based on the output we can say, that the random forest model is perfectly separating high income countries from the low income countries. However, this is might be too good to be true, and this could suggest that the test set is very similar to the training set, the data is easy to separate in the feature space, or there might be a leakage or target feature overlap.
# - But, if the model is truely perfect, then it could mean that the features used are very strong predictors of income category. As well, that the threshold (GDP per capita > usd 12,000) aligns well with the predictors. Or there is a very clear difference between hihg and low income groups in the data.
# 
# Implications
# 
# - The random forest classifier can help to understand which development indicators best distinguish high-income countries. 
# - The model can also reinforce that high income usually coexists with digital and infrastructure access. 
# - The random forest model might be able to help developing countries assess which investments, for example health, are common amoung the wealthier countries.

# ### Question 4: Classify high CO₂ emitters

# We define, in the following, high emissions as being in the top 25% of CO₂ per capita.

# In[52]:


data_q4 = data.dropna(subset=['CO2_Emissions'])

threshold = data_q4['CO2_Emissions'].quantile(0.75)
data_q4['High_CO2'] = (data_q4['CO2_Emissions'] > threshold).astype(int)

X_q4 = data_q4[['GDP_pc', 'Health_Exp_pc', 'Internet_Use', 'Power_Use', 'Renew_Energy', 'Mobile_Use']]
y_q4 = data_q4['High_CO2']

X_train, X_test, y_train, y_test = train_test_split(X_q4, y_q4, test_size=0.2, random_state=42)

rf_q4 = RandomForestClassifier(random_state=42)
rf_q4.fit(X_train, y_train)
y_pred = rf_q4.predict(X_test)

print("Classification Report - High CO2 Emissions")
print(classification_report(y_test, y_pred))


# Interpretation
# 
# - The countries with a CO2 emission per capita in the top 25% are labeled as high emmiters. 
# - The model is predicting this high emitter call by using the following development indicators: 'GDP_pc', 'Health_Exp_pc', 'Power_Use', 'Mobile_Use', and 'Renew_Energy'.
# - The higher GDP per capita and power use are strong predicotrs of hihger CO2 emissions.
# - Lower share of renewable energy is often correlates with high emissions.
# - Based on the output, the model is predicting correctly most of the time (93%).
# - The model is good at detecting both low and high CO2 emitters, with a slightly better performance on the low class.
# - The is precision of the model is high.
# - So, it can be said that the model is useful for identifying high CO2 emitters.
# 
# Implications
# 
# - The results of the model are illustrating the environmental cost of deelopment. The high-income and industrialized countries are usually emit more per capita.
# - The results of the model could guide sustainable development policies by identifying which development paths resutl in lower carbon footprint.
# - The 'Renew_Energy' can help us being a potential mitigator. 

# ### Conclusion for all questions
# 
# - Some indicators, e.g.: GDP_pc, Fertility Rate, Power Use, appear repeatedly accross the questison. This could hihglight their broad relevance to economic and social development.
# - The fertility rate and the life expectancy features can be useful subtitutes for infant mortality and public health status.
# - All the models could benefit from clean feature engineering and they could be further improved with time series features or reginoal categorical data.

# ### AUC-ROC Curves:

# In[57]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def plot_roc_curves(X, y, question_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        if name == "Logistic Regression":
            probas = model.predict_proba(X_test)[:, 1]
        else:
            probas = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {question_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[58]:


plot_roc_curves(X_q1, y_q1, "Q1: High Inequality")
plot_roc_curves(X_q2, y_q2, "Q2: Economic Instability")
plot_roc_curves(X_q3, y_q3, "Q3: High Income")
plot_roc_curves(X_q4, y_q4, "Q4: CO₂ Emissions")


# Interpretation
# 
# - The ROC (Receiver Operating Characteristic) curves plot the trade-off between the true positive rate (the sensitivity) and the false positive rate with various threshold.
# - The AUC (Area Under the Curve) quantifies the model's ability to distinguish between classes.
# 
# Implications
# 
# - The AUC scores close to 1.0 suggest a near-perfect classification performance.
# - A high AUC can show that the models can reliably distinguish between, for example, stable and unstable countries. 
# - The curves above can help to validate the robustness of the classifiers, providing a more comprehensive evaluation than accuracy alone.

# ### Decision Tree Visualization

# In[59]:


from sklearn.tree import plot_tree

def visualize_tree(X, y, question_name, max_depth=3):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['0', '1'], rounded=True)
    plt.title(f"Decision Tree - {question_name}")
    plt.show()


# In[60]:


visualize_tree(X_q1, y_q1, "Q1: High Inequality")
visualize_tree(X_q2, y_q2, "Q2: Economic Instability")
visualize_tree(X_q3, y_q3, "Q3: High Income")
visualize_tree(X_q4, y_q4, "Q4: CO₂ Emissions")


# Interpretation
# 
# - The decision tree visualization can help to reveal which features are most influental in the classification. 
# - For example, in question 3 - high-income, the top node is GDP_pc. 
# 
# Implications
# 
# - The visuals can improve the interpretability, making the models more explainable to, for example, policymakers or non-technical stakeholders.
# - The decision trees can help the governments and the NGOs to understand "what-if" scenarios, like how improving education could help the country transition to high-income status.

# ### Hyperparameter Tuning with GridSearchCV

# In[5]:


from sklearn.model_selection import GridSearchCV

def tune_model(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5]
        }
        model = RandomForestClassifier(random_state=42)
    else:
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 4, 10]
        }
        model = DecisionTreeClassifier(random_state=42)

    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    
    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)
    return grid.best_estimator_


# In[62]:


best_rf_q1 = tune_model(X_q1, y_q1, 'random_forest')


# Interpretation
# 
# - GridSearchCV is trying multiple combinations of hyperparameters to find the best-performing model.
# - For example, for a random forest, it might try different values of 'n_estimator', 'max_depth', or 'min_samples_split'.
# - It can be seen in the output, that it found, that the 'max_depth' should be 10, the 'min_samples_split' should be 2, and the 'n_estimators' should be 100. This way it gives the best score of 0.75
# 
# Implications
# 
# - It can boosts model performance by finding an optimal balance between underfitting and overfitting.
# - It can help to build a generalizable model that performs well not only on training data but also on unseen data.
# - It can be especially useful for model like in question 2, where borderline cases could lead to large social impacts.

# ### Conclusion
# 
# - The ROC curve and AUC are model quality metrics, they can verify classifier's reliability. 
# - The decision tree plots are used for interpretability, and can help to give trasnparent decision logic for stakeholders.
# - GridSearchCV is optimizing performance, and can create more stable and accurate predictive models. 
