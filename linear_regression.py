# %% [markdown]
# ## Problem Statement

# * Company: Surprise Housing (US)
# * Objective: Enter Australian housing market by flipping undervalued properties.
# * Data: CSV file containing historical Australian house sales data.
# * Task: Build a regularized regression model to:

#   - Predict actual values of prospective properties.
#   - Identify significant variables influencing price.
#   - Evaluate how well these variables describe the price.
#   - Determine optimal lambda values for ridge and lasso regression.
#   - This model will inform investment decisions by helping the company identify undervalued properties and maximize profit.
# %% [markdown]
# ## Business Goal
# - Model the price of houses with the available independent variables.
# - Use the model to understand how exactly the prices vary with the variables.
# - Manipulate the strategy of the firm and concentrate on areas that will yield high returns based on the model.
# - Understand the pricing dynamics of a new market using the model.

# %% [markdown]

# ## Data Preparation

# %%

from scipy.stats import chi2_contingency
from matplotlib import rcParams
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import f_oneway

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.feature_selection as feature_selection
import sklearn.pipeline as pipeline
import ast
import re
import json
import warnings
warnings.filterwarnings('ignore')

# %%

data_dictionary_file = "data_description.txt"
with open(data_dictionary_file) as f:
    string_data = f.read()

data_dictionary = {}
values = []
key, description = None, None
values = []
for line in string_data.split('\n'):
    if ':' in line and 'story' not in line:
        key, description = line.split(':')
        data_dictionary[key] = {"description": description}
        if values != []:
            data_dictionary[key]["values"] = '\n'.join(values)
        values = []
    else:
        values.append(line.replace('\n', ''))

data_dictionary.keys()
# %%

filename = "train.csv"

df = pd.read_csv(filename)

df

# %% [markdown]

# ## Data Understanding preparation and EDA
# %%

df.shape
num_rows = df.shape[0]
num_columns = df.shape[1]
print("number of rows: ", num_rows)
print("number of columns: ", num_columns)

# %%
# check the data types of the columns
df.info()

# %% [markdown]

# ## Data Cleaning

# %%

# checking duplicate rows
duplicate_rows = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows.shape[0])
# %%

# Set the threshold for missing values (e.g., 30% missing values)
threshold = 0.8


def drop_columns_with_missing_values(df, threshold):
    # Calculate the percentage of missing values in each column
    missing_values = df.isnull().mean()

    # Select columns where the percentage of missing values is above the threshold
    columns_to_drop = missing_values[missing_values > threshold].index.tolist()

    # Drop columns with more missing values
    df1 = df.drop(columns=columns_to_drop)

    print("dropped columns with more than {}% missing values: {}".format(
        threshold * 100, columns_to_drop))
    return df1


df = drop_columns_with_missing_values(df, threshold)

# %%

threshold = 0.8  # Define the threshold for bias (80% or more)


def drop_biased_columns(df, threshold):
    # Calculate the percentage of the most frequent value in each column
    most_frequent_value_percentage = df.apply(
        lambda col: col.value_counts(normalize=True).max())

    # Select columns where the most frequent value percentage is above the threshold
    biased_columns = most_frequent_value_percentage[most_frequent_value_percentage >= threshold].index.tolist(
    )

    # Drop columns biased towards a single/few values
    df1 = df.drop(columns=biased_columns)

    print("dropped columns biased towards a single/few values: {}".format(biased_columns))
    return df1


df = drop_biased_columns(df, threshold)

# %%


def remove_outliers(df, columns, threshold=3):
    """
    Remove outliers from specified columns in a DataFrame using z-score method.

    Parameters:
    df (pandas DataFrame): Input DataFrame.
    columns (list): List of columns to check for outliers.
    threshold (float): Threshold for outliers in terms of z-score. Default is 3.

    Returns:
    pandas DataFrame: DataFrame with outliers removed.
    """
    df_out = df.copy()
    for col in columns:
        # Calculate z-scores for the specified column
        z = np.abs(stats.zscore(df_out[col]))

        # Remove rows where z-score exceeds the defined threshold
        df_out = df_out[(z < threshold)]

    return df_out


def remove_outliers_iqr(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


# %%
# identify the columns with null values
null_value_columns = list(df.columns[df.isnull().sum() > 0])

print("there are {} columns with null values".format(len(null_value_columns)))
print("columns with null values: ", null_value_columns)


# %%

# number of null values in each column
df[null_value_columns].isnull().sum()

# %%
# percentage of null values in each column
df[null_value_columns].isnull().sum() / num_rows * 100

# %%
df[null_value_columns].describe()

# %%
null_values_numerical_columns = df[null_value_columns].describe().columns
null_values_categorical_columns = list(
    set(null_value_columns) - set(null_values_numerical_columns))

df[null_values_numerical_columns].describe()

# %%

# replace null values with median values

for null_values_numerical_column in null_values_numerical_columns:
    median = df[null_values_numerical_column].quantile([.5]).iloc[0]
    print("Replacing null values of {} with median {}".format(
        null_values_numerical_column, median))
    df[null_values_numerical_column].fillna(median, inplace=True)


# %%
# number of numerical variables
num_numerical_variables = len(df.describe().columns)
numerical_variables = list(df.describe().columns)

print("number of numerical variables: ", num_numerical_variables)
print("numerical variables: ", numerical_variables)


# %%
# find categorical variables

def find_categorical_variables(df, num_categories=50):
    categorical_variables = []
    for column in df.columns:
        if len(df[column].value_counts()) < num_categories:
            # print(column)
            categorical_variables.append(column)

    categorical_variables.sort()
    return categorical_variables


s = set()
for num_categories in range(30, 40, 5):
    categorical_variables = find_categorical_variables(df, num_categories)
    print("number of categorical variables with less than {} categories: {}".format(
        num_categories, len(categorical_variables)))
    print("categorical variables with less than {} categories: {}".format(
        num_categories, categorical_variables))
    print()
    new_categories = list(set(categorical_variables) - s)
    new_categories.sort()
    print("new categories: ", new_categories)
    s = s.union(set(categorical_variables))

# %%

for categorical_variable in categorical_variables:
    print(df[categorical_variable].value_counts())
    print()

# %%

for numerical_variable in numerical_variables:
    sns.distplot(df[numerical_variable])
    plt.show()


# %%

for null_values_categorical_column in null_values_categorical_columns:
    mode = df[null_values_categorical_column].mode()[0]
    print("mode of {}: {}".format(null_values_categorical_column, mode))
    display(df[null_values_categorical_column].value_counts())
    display(df[null_values_categorical_column].isnull().sum())
    replacement_value = "None"
    print("replacing null values of {} with {}".format(
        null_values_categorical_column, replacement_value))
    df[null_values_categorical_column].fillna(replacement_value, inplace=True)
# %% [markdown]

# ## Data Preparation

# %%


df1 = df.drop(columns=['Id'])

# convert categorical variables to categorical type
for categorical_variable in categorical_variables:
    df1[categorical_variable] = pd.Categorical(df1[categorical_variable])

len(categorical_variables)

# %%


def categorical_vs_target_boxplot(df1, target):

    range_elements = range(1, 9)
    length = len(range_elements)

    for i in range(0, len(categorical_variables), length):
        plt.figure(figsize=(25, 25))
        for k in range_elements:
            cat_variable_num = i + k - 1
            # print(cat_variable_num)
            plt.subplot(4, 2, k)
            try:
                sns.boxplot(
                    x=categorical_variables[cat_variable_num], y=target, data=df1)
                plt.title(
                    categorical_variables[cat_variable_num] + f' vs {target}')
            except Exception as e:
                print(e)
        plt.show()


categorical_vs_target_boxplot(df1, 'SalePrice')

# %%


def categorical_vs_target_barplot(df1, categorical_variable, target):
    # Grouping by 'categorical_variable' and calculating mean SalePrice, then sorting values
    mean_prices = df1.groupby(categorical_variable)[
        target].mean().sort_values(ascending=False)
    # print(mean_prices)
    # Creating a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_prices.index, y=mean_prices.values,
                palette='viridis', order=mean_prices.index)

    # Annotating mean values on top of each bar
    for i, value in enumerate(mean_prices):
        plt.text(i, value, f'{value:.2f}',
                 ha='center', va='bottom', rotation=45)

    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
    plt.xlabel(categorical_variable)
    plt.ylabel(f'Mean {target}')
    plt.title(f'Mean {target} by {categorical_variable}')
    plt.tight_layout()
    plt.show()
    if categorical_variable in data_dictionary:
        print(categorical_variable, data_dictionary[categorical_variable])
    else:
        print("No description found for {} in data dictionary".format(
            categorical_variable))


for categorical_variable in categorical_variables:
    print(categorical_variable)
    categorical_vs_target_barplot(df1, categorical_variable, 'SalePrice')
    print("\n\n")

# %%

# some categorical variables identified were actually numerical variables
# convert them to numerical variables

# numerical_variables = numerical_variables + ['3SsnPorch', 'LowQualFinSF', 'PoolArea']
# categorical_variables = list(set(categorical_variables) - set(['3SsnPorch', 'LowQualFinSF', 'PoolArea']))


# %%


# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = df

# Assuming 'categorical_var' is the column name of your categorical variable

num_features_to_rfe_info_mapping = {}
categoricals = []
f_statistics = []
p_values = []
for categorical_var in categorical_variables:
    # Assuming 'numerical_var' is the column name of your numerical variable
    numerical_var = 'SalePrice'

    # Perform ANOVA
    grouped_data = [data[numerical_var][data[categorical_var] == category]
                    for category in data[categorical_var].unique()]
    f_stat, p_value = f_oneway(*grouped_data)

    # Output the results
    # print(f" {categorical_var} F-statistic: {f_stat} P-value: {p_value}")
    categoricals.append(categorical_var)
    f_statistics.append(f_stat)
    p_values.append(p_value)

num_features_to_rfe_info_mapping = {
    "categorical": categoricals, "f_statistic": f_statistics, "p_value": p_values}
f_st_df = pd.DataFrame(num_features_to_rfe_info_mapping).sort_values(
    by=["f_statistic"], ascending=True)

sorted_by_f_st = f_st_df[f_st_df["p_value"] < 0.05].sort_values(
    by=["f_statistic"], ascending=False)
sorted_by_f_st

# %%

corr = df[numerical_variables].corr()

# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(30, 30))

# heatmap
sns.heatmap(corr, cmap="YlGnBu", annot=True)
plt.show()

# %%

corr1 = corr['SalePrice'].sort_values(ascending=False)

print("Negative Correlations:")
display(corr1[corr1 < 0])

threshold = 0.5
print("Top 10 Positive Correlations greater than {}:".format(threshold))
display(corr1[corr1 > threshold])

indices = corr1[corr1 > threshold].index


def highlight(corr, threshold_low, threshold_high, color):
    # Apply styling to the correlation matrix 'corr'
    return corr.style.apply(
        # Lambda function to apply background color 'red' if value 'v' in row 'x' is greater than 'threshold', else no styling
        lambda x: [f"background: {color}" if v >
                   threshold_low and v < threshold_high else "" for v in x],
        axis=1  # Apply the function along rows (axis=1)
    )


display(highlight(corr.loc[indices, indices], .8, 1, 'red'))
highlight(corr.loc[indices, indices], .7, .8, 'yellow')

# %%
sns.scatterplot(x='OverallQual', y='SalePrice', data=df)


# %%

sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)

# %%

df = remove_outliers(df, ['GrLivArea', 'SalePrice'])
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)


# %%


def get_metrics(y_train, y_train_pred, y_test, y_test_pred):
    training_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    training_r2 = metrics.r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    training_rss = np.sum(np.square(y_train - y_train_pred))
    test_rss = np.sum(np.square(y_test - y_test_pred))

    d = {"r2": [training_r2, test_r2], "rmse": [
        training_rmse, test_rmse], "rss": [training_rss, test_rss]}

    return pd.DataFrame(d, index=["training", "test"])


# %%

# build a model with only numerical variables
df1 = df.drop(columns=categorical_variables)
df1 = df1.drop(columns=['Id'])
X = df1.drop(columns=['SalePrice'])
y = df1['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create scaler object
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# Scale the features
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

model = Lasso(alpha=0.1)


def train_model(X_train, y_train, X_test, y_test, model):

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, get_metrics(y_train, y_train_pred, y_test, y_test_pred)


model, metrics_df = train_model(
    X_train_scaled, y_train, X_test_scaled, y_test, model)

metrics_df

# %%

df['SalePrice'].describe()
# %%


pd.Series(model.coef_, X_train.columns).sort_values(ascending=False)

# %%
# sorted_by_f_st.head(2)
important_categoricals = sorted_by_f_st[sorted_by_f_st['f_statistic'] > 300]
display(important_categoricals)
important_categoricals = important_categoricals['categorical'].tolist()

for important_categorical in important_categoricals:
    print(important_categorical)
    # categorical_vs_target_barplot(df1, important_categorical, 'SalePrice')
    df[important_categorical].value_counts().plot(kind='bar')
    plt.show()


print(important_categoricals)

# %%

sorted_by_f_st


def sort_categorical_variables_by_independence(df, categorical_vars):
    independence_scores = []

    # Iterate through pairs of categorical variables
    for i in range(len(categorical_vars)):
        for j in range(i + 1, len(categorical_vars)):
            cross_tab = pd.crosstab(
                df[categorical_vars[i]], df[categorical_vars[j]])
            chi2, _, _, _ = chi2_contingency(cross_tab)
            n = cross_tab.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))
            independence_scores.append(
                (categorical_vars[i], categorical_vars[j], cramers_v))

    # Create DataFrame of independence scores and sort by Cramer's V
    sorted_df = pd.DataFrame(independence_scores, columns=[
                             'Variable 1', 'Variable 2', 'Cramer\'s V'])
    sorted_df = sorted_df.sort_values(by='Cramer\'s V', ascending=False)

    return sorted_df


sorted_df = sort_categorical_variables_by_independence(
    df, sorted_by_f_st['categorical'].tolist())

sorted_df

# %%

sorted_df[0:50]

# %%

len(categorical_variables)

# %%


def create_dummy_variables(df, categorical_variables):
    """
    Create dummy variables for categorical variables in a DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame containing categorical variables.
    categorical_variables (list): List of categorical variables.

    Returns:
    pandas DataFrame: DataFrame with dummy variables for categorical variables.
    """
    dummies = []
    for categorical_variable in categorical_variables:
        dummy = pd.get_dummies(
            df[categorical_variable], prefix=categorical_variable, drop_first=True)
        dummies.append(dummy)

    df = df.drop(categorical_variables, axis=1)
    df = pd.concat([df] + dummies, axis=1)

    return df


def perform_rfe_and_get_summary(X_train, y_train, n_features=10):
    """
    Perform Recursive Feature Elimination (RFE) using Linear Regression and provide a summary.

    Parameters:
    X_train (pandas DataFrame or array-like): Training features.
    y_train (pandas Series or array-like): Training target.
    n_features (int): Number of features to select. Default is 10.

    Returns:
    tuple: Tuple containing RFE object and DataFrame summarizing the selection process.
    """
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    rfe_df = pd.DataFrame({
        "column": X_train.columns,
        "support": rfe.support_,
        "ranking": rfe.ranking_
    })
    rfe_df.sort_values(by="ranking", inplace=True)

    return rfe, rfe_df


def create_X_y(df, target_variable):
    """
    Create features (X) and target variable (y) from a DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame containing features and target variable.
    target_variable (str): Name of the target variable.

    Returns:
    tuple: Tuple containing features (X) and target variable (y).
    """
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    return X, y


def perform_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Perform train-test split for features and target variable.

    Parameters:
    X (pandas DataFrame or array-like): Features.
    y (pandas Series or array-like): Target variable.
    test_size (float or int): Proportion of the dataset to include in the test split.
                              Default is 0.3 (30% for the test set).
    random_state (int): Controls the shuffling applied to the data before splitting.
                        Pass an int for reproducible output. Default is 42.

    Returns:
    tuple: Tuple containing X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def scale_features_with_scaler(scaler, X_train, X_test):
    """
    Scale the features using a specified scaler.

    Parameters:
    scaler (scaler object): Scaler instance (e.g., StandardScaler, MinMaxScaler, etc.).
    X_train (pandas DataFrame or array-like): Training features.
    X_test (pandas DataFrame or array-like): Test features.

    Returns:
    tuple: Tuple containing scaled X_train and X_test.
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def scaled_arrays_to_dataframe(X_scaled, columns):
    """
    Convert scaled NumPy array to a pandas DataFrame with specified columns.

    Parameters:
    X_scaled (array-like): Scaled feature array.
    columns (list): Column names for the DataFrame.

    Returns:
    pandas DataFrame: DataFrame containing scaled features.
    """
    df_scaled = pd.DataFrame(X_scaled, columns=columns)
    return df_scaled

# %%


def experiment_num_features(n):
    df2 = create_dummy_variables(df, categorical_variables)

    X, y = create_X_y(df2, 'SalePrice')
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_features_with_scaler(
        MinMaxScaler(), X_train, X_test)
    X_train_scaled = scaled_arrays_to_dataframe(
        X_train_scaled, X_train.columns)
    X_test_scaled = scaled_arrays_to_dataframe(X_test_scaled, X_test.columns)
    rfe, rfe_df = perform_rfe_and_get_summary(
        X_train_scaled, y_train, n_features=n)

    rfe_df1 = rfe_df[0:n]["column"].apply(
        lambda x: x.split("_")[0]).value_counts()

    return rfe, rfe_df, rfe_df1


num_features_to_rfe_info_mapping = {}
for i in range(50, 151, 10):
    rfe, rfe_df, rfe_df1 = experiment_num_features(i)
    num_features_to_rfe_info_mapping[i] = (rfe, rfe_df, rfe_df1)

# %%
num_features_to_rfe_info_mapping[50][2].index.tolist()
# %%


def create_model_with_num_features(num_features):
    df1 = df[num_features_to_rfe_info_mapping[num_features]
             [2].index.tolist() + ['SalePrice']]
    print("actual number of independent features: ", len(df1.columns) - 1)
    cat = list(set(categorical_variables) & set(df1.columns))
    df2 = create_dummy_variables(df1, cat)

    X, y = create_X_y(df2, 'SalePrice')
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_features_with_scaler(
        MinMaxScaler(), X_train, X_test)
    X_train_scaled = scaled_arrays_to_dataframe(
        X_train_scaled, X_train.columns)
    X_test_scaled = scaled_arrays_to_dataframe(X_test_scaled, X_test.columns)

    model = Lasso(alpha=0.1)
    model, metrics_df = train_model(
        X_train_scaled, y_train, X_test_scaled, y_test, model)
    print(num_features, len(df1.columns), df1.columns)
    display(metrics_df)
    return model, metrics_df, X_train_scaled


for num_features in num_features_to_rfe_info_mapping:
    model, metrics_df, X_train_scaled = create_model_with_num_features(
        num_features)

# %%


def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for features in a DataFrame.

    Parameters:
    X (pandas DataFrame): DataFrame containing the features.

    Returns:
    pandas DataFrame: DataFrame with calculated VIF for each feature.
    """
    vif_df = pd.DataFrame()
    vif_df['Features'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    vif_df['VIF'] = round(vif_df['VIF'], 2)
    vif_df = vif_df.sort_values(by="VIF", ascending=False)
    return vif_df


def build_ols_model(X, y):
    """
    Build an Ordinary Least Squares (OLS) model using statsmodels, print the model summary, and return the fitted model.

    Parameters:
    X (pandas DataFrame): DataFrame containing the features.
    y (pandas Series): Series containing the target variable.

    Returns:
    statsmodels regression summary: Summary of the fitted OLS model.
    """
    X = sm.add_constant(X)  # Adding a constant column for intercept
    results = sm.OLS(y.values, X).fit()  # Fit the OLS model
    print(results.summary())  # Print the summary of the fitted model
    return results  # Return the fitted model


def build_ols_lasso_model(X, y):
    """
    Build an Ordinary Least Squares (OLS) model using statsmodels, print the model summary, and return the fitted model.

    Parameters:
    X (pandas DataFrame): DataFrame containing the features.
    y (pandas Series): Series containing the target variable.

    Returns:
    statsmodels regression summary: Summary of the fitted OLS model.
    """
    X = sm.add_constant(X)  # Adding a constant column for intercept
    results = sm.OLS(y.values, X).fit_regularized(
        method='sqrt_lasso')  # Fit the OLS model
    try:
        print(results.summary())  # Print the summary of the fitted model
    except Exception as e:
        print(e)
    return results  # Return the fitted model


def drop_features_and_create_model(df, features_to_drop, target_variable, model_builder=build_ols_model):
    df1 = df[num_features_to_rfe_info_mapping[num_features]
             [2].index.tolist() + [target_variable]]

    df1 = df1.drop(columns=features_to_drop)
    print("actual number of independent features: ", len(df1.columns) - 1)
    cat = list(set(categorical_variables) & set(df1.columns))
    df2 = create_dummy_variables(df1, cat)

    X, y = create_X_y(df2, 'SalePrice')
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = scale_features_with_scaler(
        MinMaxScaler(), X_train, X_test)
    X_train_scaled = scaled_arrays_to_dataframe(
        X_train_scaled, X_train.columns)
    X_test_scaled = scaled_arrays_to_dataframe(X_test_scaled, X_test.columns)

    vif_df = calculate_vif(X_train_scaled)

    display(vif_df)
    variables_with_infinite_vif = vif_df[vif_df['VIF'] == np.inf]["Features"].apply(
        lambda x: x.split("_")[0]).value_counts().index.tolist()

    print("variables with infinite VIF: ", variables_with_infinite_vif)
    return model_builder(X_train_scaled, y_train), (X_train_scaled, X_test_scaled, y_train, y_test)


def get_summary_info_sorted(model):
    """
    Extract feature names, coefficients, t-values, and p-values from the summary of an OLS model and sort by p-values.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted OLS model.

    Returns:
    pandas.DataFrame: DataFrame containing information for each feature sorted by p-values.
    """
    # Get the summary of the fitted model
    summary = model.summary()

    # Extract information from the summary
    results = summary.tables[1].data

    # Create DataFrame with extracted information
    columns = ["feature", "coef", "std err", "t", "p", "[0.025", "0.975]"]
    df = pd.DataFrame(results)

    # display(df)

    new_header = df.iloc[0]  # Grab the first row for the header
    new_header[0] = "feature"
    df = df[1:]  # Take the data except the first row
    df.columns = new_header  # Set the header row as the DataFrame column names
    # print("columns are", df.columns)
    # Convert relevant columns to numeric types
    p_column = 'P>|t|'
    df[['coef', 't', p_column]] = df[[
        'coef', 't', p_column]].apply(pd.to_numeric)

    # # Sort DataFrame by p-values
    sorted_df = df.sort_values(by=p_column, ascending=False)

    return sorted_df


def get_r2_adjusted_r2(model):
    """
    Extract R-squared and adjusted R-squared values from the summary of a fitted model.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted regression model.

    Returns:
    tuple: Tuple containing R-squared and adjusted R-squared values.
    """
    # Get the summary of the fitted model
    summary = model.summary()

    # Extract R-squared and adjusted R-squared from the summary
    r_squared = float(summary.tables[0].data[0][3])
    adj_r_squared = float(summary.tables[0].data[1][3])

    print("R-squared: ", r_squared, "Adjusted R-squared: ", adj_r_squared)
    return r_squared, adj_r_squared


# %%
for num_features in num_features_to_rfe_info_mapping:
    model, metrics_df, X_train_scaled = create_model_with_num_features(
        num_features)

num_features = 50
model, metrics_df, X_train_scaled = create_model_with_num_features(
    num_features)

# %%
vif_df = calculate_vif(X_train_scaled)

variables_with_infinite_vif = vif_df[vif_df['VIF'] == np.inf]["Features"].apply(
    lambda x: x.split("_")[0]).value_counts().index.tolist()

print("variables with infinite VIF: ", variables_with_infinite_vif)


build_ols_model(X_train_scaled, y_train)

# %%
sorted_df[sorted_df['Variable 2'] == 'BedroomAbvGr'].head(1)

##

# %%
columns_to_drop = []
# %%
#


model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)

df3[['feature', 'P>|t|']].head(20)


# %%
columns_to_drop.extend(['LotConfig'])
print("Dropped columns: ", columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)


# %%
columns_to_drop.extend(['GarageType'])
print("Dropped columns: ", columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%
columns_to_drop.extend(['MSSubClass'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['Exterior2nd'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['Foundation'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%
columns_to_drop.extend(['FireplaceQu'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%
columns_to_drop.extend(['Neighborhood'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['Exterior1st'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['TotRmsAbvGrd'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['BsmtFullBath'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)


# %%

columns_to_drop.extend(['OverallQual'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['GarageCars'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['BsmtFinType1'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['1stFlrSF'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['2ndFlrSF'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)

# %%

columns_to_drop.extend(['OverallCond'])
print(f"Dropped the following columns: ")
display(columns_to_drop)
model, data = drop_features_and_create_model(df, columns_to_drop, 'SalePrice')
df3 = get_summary_info_sorted(model)
get_r2_adjusted_r2(model)
df3[['feature', 'P>|t|']].head(20)


# %%

columns_to_drop1 = []
print(f"Dropped the following columns: ")
display(columns_to_drop1)
model, data = drop_features_and_create_model(
    df, columns_to_drop1, 'SalePrice', build_ols_lasso_model)
# df3 = get_summary_info_sorted(model)
# get_r2_adjusted_r2(model)
# df3[['feature', 'P>|t|']].head(20)
