import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
from Helper_Funcs.EDA_funcs import check_df

import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


########################
# Loading the data
########################

train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date'])
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()

check_df(train)
check_df(test)
#check_df(sample_sub)
check_df(df)
# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

########################
# Date Features
########################

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)

    df['quarter_of_year'] = df.date.dt.quarter
    df['quarter_of_year'] = df.date.dt.quarter
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['is_year_start'] = df.date.dt.is_year_start.astype(int)
    df['is_year_end'] = df.date.dt.is_year_end.astype(int)
    df['daysinmonth'] = df.date.dt.daysinmonth
    return df

df = create_date_features(df)

########################
# Random Noise
########################
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))

########################
# Lag/Shifted Features
########################


df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 271, 364, 546, 728])

########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df,[91,182,364,546,728])


#######################################
# Exponentially Weighted Mean Features
#######################################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)



########################
# One-Hot Encoding
########################
# Eger view cok fazla olursa, rare ile gruplara bölü azalt sonra  one hot encoding yap
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

################################################
# Converting sales to log(1+sales)
################################################

df['sales'] = np.log1p(df["sales"].values)
check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################

# 2017 as ending point for train set
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017 first 3 months as val set
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

########################
# LightGBM Model
########################


# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthrea d': -1}



lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

print("SNAPE:",smape(np.expm1(y_pred_val), np.expm1(Y_val))) # 13.4225
print('RMSE:', np.sqrt(mean_squared_error(np.expm1(Y_val), np.expm1(y_pred_val)))) # 6.89
print("MAE:",mean_absolute_error(np.expm1(Y_val), np.expm1(y_pred_val))) # 5.32


def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:num])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))



plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)



########################
# Final Model
########################

train = df.loc[~df.sales.isna()] # df.loc[df.sales.notnul()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

gb_params = {'metric': 'mae',
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1, # n_jobs
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)


model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)