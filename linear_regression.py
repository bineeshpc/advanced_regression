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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

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
for num_categories in range(5, 100, 30):
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

df1 = df.drop(columns=['Id'])

#%%

# X = df1.drop(columns=['SalePrice'])
# y = df1['SalePrice']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = Lasso(alpha=0.1)
# model.fit(X_train, y_train)
# model.score(X_test, y_test)