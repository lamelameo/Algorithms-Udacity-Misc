''' Kaggle Machine Learning course practice stuff '''

# --------------- IMPORTING MODULES/PACKAGES -----------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

housedata = pd.read_csv(r'train.csv')

# ---------------------- SETTING UP DATA FOR USE --------------------------------------

housedata.dropna(axis=0, subset=['SalePrice'], inplace=True)  # drop rows with no saleprice
X_predictors = housedata.drop(['SalePrice'], axis=1)  # drop saleprice column from predictors
target = housedata.SalePrice

candidate_train_predictors = housedata.drop(['Id', 'SalePrice'], axis=1)
# not dropping any other values yet as going to split to impute
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                               #candidate_train_predictors[cname].nunique() < 10 and
                               candidate_train_predictors[cname].dtype == 'object']
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
#print('numeric cols:', numeric_cols, '\n')

# splitting data to impute only numeric stuff
train_numeric = candidate_train_predictors[numeric_cols]
train_categorical = candidate_train_predictors[low_cardinality_cols]
print('numeric shape:', train_numeric.shape, 'categoric shape:', train_categorical.shape, '\n')
# 36 columns numeric, 43 columns categoric

# ------------------------ LOOKING AT DATA ------------------------------------

print(housedata.columns, '\n')
rel_num_X = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF']
rel_cat_X = ['TotRmsAbvGrd', 'BedroomAbvGr', 'FullBath']
print(target.describe(), '\n')
# historgram of saleprice:
sns.distplot(target)  # positive skew not normally distributed
print('skewness:{}'.format(target.skew()))
print('Kurtosis:{}'.format(target.kurt()), '\n')

def plotstuff():
    for variable in rel_num_X:
        var = variable
        data = pd.concat([target, housedata[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', s=1, ylim=(0, 800000))
    for variable in rel_cat_X:
        var = variable
        data = pd.concat([target, housedata[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)

    var = 'OverallQual'
    data = pd.concat([target, housedata[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data, fliersize=1)
    fig.axis(ymin=0, ymax=800000)

    #var = 'YearBuilt'
    #data = pd.concat([target, housedata[var]], axis=1)
    #f, ax = plt.subplots(figsize=(16, 8))
    #fig = sns.boxplot(x=var, y="SalePrice", data=data)
    #fig.axis(ymin=0, ymax=800000)
    #plt.xticks(rotation=90)


#correlation matrix
corrmat = housedata.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True)

# sale price correlation matrix
k = 15  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(housedata[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)

# clustermap
#data1 = housedata.dropna(axis=1)
corr = housedata.corr()
clustermap = sns.clustermap(corr)

plt.show()

# remove after done with section -------------->@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
quit()

# ------------------------LOOKING FOR MISSING VALUES, FILLING/DROPPING -----------------------------------

print('train categorical missing values:\n', train_categorical.isnull().sum(), '\n')
# dataframe of categorical data has 15 columns with missing values, some with almost all missing values
# should drop all columns or just those with almost no data or leave all??
cols_missing = [col for col in train_categorical.columns
                if train_categorical[col].isnull().any()]
reduced_train_cat = train_categorical.drop(cols_missing, axis=1)  # dropped all with missing values

print('train numeric missing values:', train_numeric.isnull().sum(), '\n')
# dataframe of numeric data has missing values in three columns: LotFrontage, MasVnrArea, GarageYrBlt
# makes no sense to impute values for some of these as maybe means None instead?
print('num cols with missing values:\n', train_numeric[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].describe(), '\n')
fill_cols = {'MasVnrArea': 0, 'LotFrontage': 0}  # fill only these columns - cant fill with 'None'
numeric_fill = train_numeric.fillna(value=fill_cols)
print('num fill:\n', numeric_fill[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].describe(), '\n')

# stats and histograms:
figure1 = plt.figure(1)
numeric_fill['MasVnrArea'].plot.hist(bins=20)
print('MasVnrArea\n', 'mode:', numeric_fill['MasVnrArea'].mode(), 'median:', numeric_fill['MasVnrArea'].median(),
      'mean:', numeric_fill['MasVnrArea'].mean())
figure2 = plt.figure(2)
numeric_fill['LotFrontage'].plot.hist(bins=20)
print('LotFrontage\n', 'mode:', numeric_fill['LotFrontage'].mode(), 'median:', numeric_fill['LotFrontage'].median(),
      'mean:', numeric_fill['LotFrontage'].mean())
figure3 = plt.figure(3)
numeric_fill['GarageYrBlt'].plot.hist(bins=20)
print('GarageYrBlt\n', 'mode:', numeric_fill['GarageYrBlt'].mode(), 'median:', numeric_fill['GarageYrBlt'].median(),
      'mean:', numeric_fill['GarageYrBlt'].mean())
print()

#plt.show()

# --------------IMPUTE NUMERIC DATA --- OHE CATEGORICAL ----------------------
# impute:
my_imputer = Imputer()
imputed_numeric_train = my_imputer.fit_transform(train_numeric)
# convert numpy array to dataframe for merging
df_impute_numeric_train = pd.DataFrame(imputed_numeric_train, columns=train_numeric.columns)
print('imputed numeric data:\n', df_impute_numeric_train.shape, '\n')  # added index by 'columns='

# one hot encode:
encoded_categoric_train = pd.get_dummies(reduced_train_cat)
print('encoded categorical data:\n', encoded_categoric_train.shape, '\n')
#183 columns of data

def corrmat():
    saleprice = pd.DataFrame(target)
    cat_merge = encoded_categoric_train.merge(saleprice, right_index=True, left_index=True)
    num_merge = df_impute_numeric_train.merge(saleprice, right_index=True, left_index=True)
    print(cat_merge.shape)
    print(num_merge.shape)

    #correlation matrix
    corrmat1 = cat_merge.corr()
    corrmat2 = num_merge.corr()

    k = 13  # number of variables for heatmap
    cols = corrmat1.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(cat_merge[cols].values.T)
    sns.set(font_scale=1)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5},
                     yticklabels=cols.values, xticklabels=cols.values)

    # clustermaps
    #sns.clustermap(corrmat1)
    #sns.clustermap(corrmat2)
    plt.show()
#corrmat()

# -------------------- MERGE AND DETERMINE MAE------------------------------------------------

# merge imputed numeric data and one hot encoded categoric data
train_num_cat_merge = encoded_categoric_train.merge(df_impute_numeric_train, right_index=True, left_index=True)
print('num/cat merged:', train_num_cat_merge.shape)  # 1460 rows, 162 rows

def get_mae(X, y):
    # mutply by -1 to get positive value...
    return -1 * cross_val_score(RandomForestRegressor(50),
                               X, y,
                               scoring = 'neg_mean_absolute_error').mean()

#print('mae encoded cat train:', get_mae(encoded_categoric_train, target))
print()
#print('mae numeric imputed train:', get_mae(imputed_numeric_train, target))
print()
#print('mae numeric+categoric train:', get_mae(train_num_cat_merge, target))

# -------------------------- PARTIAL DEPENDENCE PLOTS -----------------------------

cols_to_use = ['LotArea', 'YrSold', 'YearBuilt', 'OverallCond', 'LotFrontage', 'GarageYrBlt', 'MasVnrArea']
thing = [train_num_cat_merge, df_impute_numeric_train, encoded_categoric_train]
print(df_impute_numeric_train['MasVnrArea'].describe())

def partial_dependence_plots():
    X = train_num_cat_merge
    y = target
    my_model = GradientBoostingRegressor()
    my_model.fit(X, y)
    my_plots = plot_partial_dependence(my_model, features=[5], X=X,
                                       feature_names=cols_to_use, grid_resolution=40)
    # features tells the index of the columns to be shown...
#partial_dependence_plots()
#plt.show()
