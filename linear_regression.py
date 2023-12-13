#%% [markdown]
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
#%% [markdown]
# ## Business Goal 
# - Model the price of houses with the available independent variables.
# - Use the model to understand how exactly the prices vary with the variables.
# - Manipulate the strategy of the firm and concentrate on areas that will yield high returns based on the model.
# - Understand the pricing dynamics of a new market using the model.

#%% [markdown]

# ## Data Preparation

#%%

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

#%%

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
#%%

filename = "train.csv"

df = pd.read_csv(filename)

df

#%% [markdown]

# ## Data Understanding preparation and EDA
#%%

df.shape
num_rows = df.shape[0]
num_columns = df.shape[1]
print("number of rows: ", num_rows)
print("number of columns: ", num_columns)

#%%
# check the data types of the columns
df.info()

#%% [markdown]

# ## Data Cleaning
#%%
# identify the columns with null values
null_value_columns = list(df.columns[df.isnull().sum() > 0])

print("there are {} columns with null values".format(len(null_value_columns)))
print("columns with null values: ", null_value_columns)


#%%

# number of null values in each column
df[null_value_columns].isnull().sum()

#%%
# percentage of null values in each column
df[null_value_columns].isnull().sum() / num_rows * 100

#%%
df[null_value_columns].describe()

#%%
null_values_numerical_columns = df[null_value_columns].describe().columns
null_values_categorical_columns = list(set(null_value_columns) - set(null_values_numerical_columns))

df[null_values_numerical_columns].describe()

#%%

# replace null values with median values

for null_values_numerical_column in null_values_numerical_columns:
    median = df[null_values_numerical_column].quantile([.5]).iloc[0]
    print("Replacing null values of {} with median {}".format(null_values_numerical_column, median))
    df[null_values_numerical_column].fillna(median, inplace=True)


#%%
# number of numerical variables
num_numerical_variables = len(df.describe().columns)
numerical_variables = list(df.describe().columns)   

print("number of numerical variables: ", num_numerical_variables)
print("numerical variables: ", numerical_variables)


#%%
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
    print("number of categorical variables with less than {} categories: {}".format(num_categories, len(categorical_variables)))
    print("categorical variables with less than {} categories: {}".format(num_categories, categorical_variables))
    print()
    new_categories = list(set(categorical_variables) - s)
    new_categories.sort()
    print("new categories: ", new_categories)
    s = s.union(set(categorical_variables))
    
#%%

for categorical_variable in categorical_variables:
    print(df[categorical_variable].value_counts())
    print()
    
#%%

for numerical_variable in numerical_variables:
    sns.distplot(df[numerical_variable])
    plt.show()


#%% 

for null_values_categorical_column in null_values_categorical_columns:
    mode = df[null_values_categorical_column].mode()[0]
    print("mode of {}: {}".format(null_values_categorical_column, mode))
    display(df[null_values_categorical_column].value_counts())
    display(df[null_values_categorical_column].isnull().sum())
    replacement_value = "None"
    print("replacing null values of {} with {}".format(null_values_categorical_column, replacement_value))
    df[null_values_categorical_column].fillna(replacement_value, inplace=True)
#%% [markdown]

# ## Data Preparation

#%%


df1 = df.drop(columns=['Id'])

# convert categorical variables to categorical type
for categorical_variable in categorical_variables:
    df1[categorical_variable] = pd.Categorical(df1[categorical_variable])

len(categorical_variables)

#%%
def categorical_vs_target_boxplot(df1, target):
    

    range_elements = range(1, 9)
    length = len(range_elements)
    
    for i in range(0, len(categorical_variables), length):
        plt.figure(figsize=(25, 25))
        for k in range_elements:
            cat_variable_num = i + k -1
            # print(cat_variable_num)
            plt.subplot(4, 2, k)
            try:
                sns.boxplot(x=categorical_variables[cat_variable_num], y=target, data=df1)
                plt.title(categorical_variables[cat_variable_num] + f' vs {target}')
            except Exception as e:
                print(e)
        plt.show()
        

categorical_vs_target_boxplot(df1, 'SalePrice')

#%%


def categorical_vs_target_barplot(df1, categorical_variable, target):
    # Grouping by 'categorical_variable' and calculating mean SalePrice, then sorting values
    mean_prices = df1.groupby(categorical_variable)[target].mean().sort_values(ascending=False)
    # print(mean_prices)
    # Creating a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_prices.index, y=mean_prices.values, palette='viridis', order=mean_prices.index)
    
    # Annotating mean values on top of each bar
    for i, value in enumerate(mean_prices):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', rotation=45)

    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
    plt.xlabel(categorical_variable)
    plt.ylabel(f'Mean {target}')
    plt.title(f'Mean {target} by {categorical_variable}')
    plt.tight_layout()
    plt.show()
    if categorical_variable in data_dictionary:
        print(categorical_variable, data_dictionary[categorical_variable])
    else:
        print("No description found for {} in data dictionary".format(categorical_variable))

for categorical_variable in categorical_variables:
    print(categorical_variable)
    categorical_vs_target_barplot(df1, categorical_variable, 'SalePrice')
    print("\n\n")
    
#%%

# some categorical variables identified were actually numerical variables
# convert them to numerical variables

numerical_variables = numerical_variables + ['3SsnPorch', 'LowQualFinSF', 'PoolArea']
categorical_variables = list(set(categorical_variables) - set(['3SsnPorch', 'LowQualFinSF', 'PoolArea']))


#%%



# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = df

# Assuming 'categorical_var' is the column name of your categorical variable

d = {}
categoricals = []
f_statistics = []
p_values = []
for categorical_var in categorical_variables:
    # Assuming 'numerical_var' is the column name of your numerical variable
    numerical_var = 'SalePrice'

    # Perform ANOVA
    grouped_data = [data[numerical_var][data[categorical_var] == category] for category in data[categorical_var].unique()]
    f_stat, p_value = f_oneway(*grouped_data)

    # Output the results
    # print(f" {categorical_var} F-statistic: {f_stat} P-value: {p_value}")
    categoricals.append(categorical_var)
    f_statistics.append(f_stat)
    p_values.append(p_value)
    
d = {"categorical": categoricals, "f_statistic": f_statistics, "p_value": p_values}
f_st_df = pd.DataFrame(d).sort_values(by=["f_statistic"], ascending=True)

f_st_df[f_st_df["p_value"] < 0.05].sort_values(by=["f_statistic"], ascending=False)


#%%

corr = df[numerical_variables].corr()

# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(30,30))

# heatmap
sns.heatmap(corr, cmap="YlGnBu", annot=True)
plt.show()

#%%

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
        lambda x: [f"background: {color}" if v > threshold_low and v < threshold_high else "" for v in x],
        axis=1  # Apply the function along rows (axis=1)
    )
    
display(highlight(corr.loc[indices, indices], .8, 1, 'red'))
highlight(corr.loc[indices, indices], .7, .8, 'yellow')
#%%

def get_metrics(y_train, y_train_pred, y_test, y_test_pred):
    training_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    training_r2 = metrics.r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    training_rss = np.sum(np.square(y_train - y_train_pred))
    test_rss = np.sum(np.square(y_test - y_test_pred))
    
    d = {"r2": [training_r2, test_r2], "rmse": [training_rmse, test_rmse], "rss": [training_rss, test_rss]}
    
    return pd.DataFrame(d, index=["training", "test"])
    
    
#%%

# build a model with only numerical variables
df1 = df.drop(columns=categorical_variables)
df1 = df1.drop(columns=['Id'])
X = df1.drop(columns=['SalePrice'])
y = df1['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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


model, metrics_df = train_model(X_train_scaled, y_train, X_test_scaled, y_test, model)

metrics_df

#%%

df['SalePrice'].describe()
#%%


pd.Series(model.coef_, X_train.columns).sort_values(ascending=False)

