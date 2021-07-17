#######################################
# Hitters
#######################################

# !pip install xgboost
# !pip install lightgbm
# conda install -c conda-forge lightgbm
# !pip install catboost

import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,RandomizedSearchCV
from helpers.data_prep import *
from helpers.eda import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
import math as mt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def hitters_load():
    data = pd.read_csv("datasets/hitters.csv")
    return data

df = hitters_load()

#df = pd.read_csv("datasets/hitters.csv")
#df.head()

#######################################
# Quick Data Preprocessing
#######################################

def hitters_data_prep(dataframe):
    # 1. Feature Engineering (Değişken Mühendisliği)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    for i in num_cols:
        dataframe[i] = dataframe[i].add(1)

    # RATIO OF VARİABLES

    # CAREER RUNS RATİO
    dataframe["NEW_C_RUNS_RATIO"] = dataframe["RUNS"] / dataframe["CRUNS"]
    # CAREER BAT RATİO
    dataframe["NEW_C_ATBAT_RATIO"] = dataframe["ATBAT"] / dataframe["CATBAT"]
    # CAREER HİTS RATİO
    dataframe["NEW_C_HITS_RATIO"] = dataframe["HITS"] / dataframe["CHITS"]
    # CAREER HMRUN RATİO
    dataframe["NEW_C_HMRUN_RATIO"] = dataframe["HMRUN"] / dataframe["CHMRUN"]
    # CAREER RBI RATİO
    dataframe["NEW_C_RBI_RATIO"] = dataframe["RBI"] / dataframe["CRBI"]
    # CAREER WALKS RATİO
    dataframe["NEW_C_WALKS_RATIO"] = dataframe["WALKS"] / dataframe["CWALKS"]
    dataframe["NEW_C_HIT_RATE"] = dataframe["CHITS"] / dataframe["CATBAT"]
    dataframe["NEW_C_RUNNER"] = dataframe["CRBI"] / dataframe["CHITS"]
    dataframe["NEW_C_HIT-AND-RUN"] = dataframe["CRUNS"] / dataframe["CHITS"]
    dataframe["NEW_C_HMHITS_RATIO"] = dataframe["CHMRUN"] / dataframe["CHITS"]
    dataframe["NEW_C_HMATBAT_RATIO"] = dataframe["CATBAT"] / dataframe["CHMRUN"]

    # PLayer Ratio in year
    dataframe["NEW_ASSISTS_RATIO"] = dataframe["ASSISTS"] / dataframe["ATBAT"]
    dataframe["NEW_HITS_RECALL"] = dataframe["HITS"] / (dataframe["HITS"] + dataframe["ERRORS"])
    dataframe["NEW_NET_HELPFUL_ERROR"] = (dataframe["WALKS"] - dataframe["ERRORS"]) / dataframe["WALKS"]
    dataframe["NEW_TOTAL_SCORE"] = (dataframe["RBI"] + dataframe["ASSISTS"] + dataframe["WALKS"] - dataframe["ERRORS"]) / dataframe["ATBAT"]
    dataframe["NEW_HIT_RATE"] = dataframe["HITS"] / dataframe["ATBAT"]
    dataframe["NEW_TOUCHER"] = dataframe["ASSISTS"] / dataframe["PUTOUTS"]
    dataframe["NEW_RUNNER"] = dataframe["RBI"] / dataframe["HITS"]
    dataframe["NEW_HIT-AND-RUN"] = dataframe["RUNS"] / (dataframe["HITS"])
    dataframe["NEW_HMHITS_RATIO"] = dataframe["HMRUN"] / dataframe["HITS"]
    dataframe["NEW_HMATBAT_RATIO"] = dataframe["ATBAT"] / dataframe["HMRUN"]
    dataframe["NEW_TOTAL_CHANCES"] = dataframe["ERRORS"] + dataframe["PUTOUTS"] + dataframe["ASSISTS"]

    # Annual Averages- Powered by Ahmet Hoca
    dataframe["NEW_CATBAT_MEAN"] = dataframe["CATBAT"] / dataframe["YEARS"]
    dataframe["NEW_CHITS_MEAN"] = dataframe["CHITS"] / dataframe["YEARS"]
    dataframe["NEW_CHMRUN_MEAN"] = dataframe["CHMRUN"] / dataframe["YEARS"]
    dataframe["NEW_CRUNS_MEAN"] = dataframe["CRUNS"] / dataframe["YEARS"]
    dataframe["NEW_CRBI_MEAN"] = dataframe["CRBI"] / dataframe["YEARS"]
    dataframe["NEW_CWALKS_MEAN"] = dataframe["CWALKS"] / dataframe["YEARS"]

    # PLAYER YEARS LEVEL
    dataframe.loc[(dataframe["YEARS"] <= 2), "NEW_YEARS_LEVEL"] = "Junior"
    dataframe.loc[(dataframe["YEARS"] > 2) & (dataframe['YEARS'] <= 5), "NEW_YEARS_LEVEL"] = "Mid"
    dataframe.loc[(dataframe["YEARS"] > 5) & (dataframe['YEARS'] <= 10), "NEW_YEARS_LEVEL"] = "Senior"
    dataframe.loc[(dataframe["YEARS"] > 10), "NEW_YEARS_LEVEL"] = "Expert"

    # PLAYER YEARS LEVEL X DIVISION

    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Junior") & (dataframe["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Junior") & (dataframe["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Mid") & (dataframe["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Mid") & (dataframe["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Senior") & (dataframe["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Senior") & (dataframe["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Expert") & (dataframe["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
    dataframe.loc[(dataframe["NEW_YEARS_LEVEL"] == "Expert") & (dataframe["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

    # Player Promotion to Next League
    dataframe.loc[(dataframe["LEAGUE"] == "N") & (dataframe["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
    dataframe.loc[(dataframe["LEAGUE"] == "A") & (dataframe["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
    dataframe.loc[(dataframe["LEAGUE"] == "N") & (dataframe["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
    dataframe.loc[(dataframe["LEAGUE"] == "A") & (dataframe["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # 2. Outliers (Aykırı Değerler)

    for col in num_cols:
        replace_with_thresholds(dataframe, col)



    # dataframe["SALARY"] = dataframe["SALARY"].fillna(dataframe.groupby(["NEW_YEARS_LEVEL", "DIVISION", "LEAGUE"])["SALARY"].transform("median"))

    # 4. Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    ylevel = ['Junior', 'Mid', 'Senior', 'Expert']
    ordi = OrdinalEncoder(categories=[ylevel])
    dataframe["NEW_YEARS_LEVEL"] = ordi.fit_transform(dataframe[["NEW_YEARS_LEVEL"]])
    dataframe["NEW_YEARS_LEVEL"] = dataframe["NEW_YEARS_LEVEL"].astype(int)

    # 5. Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01)

    # 6. One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols,drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # 3.missing values
    # Powered by Ahmet Hoca - all rights reserved to cem hoca
    imputer = KNNImputer(n_neighbors=5)
    dataframe_filled = imputer.fit_transform(dataframe)
    dataframe = pd.DataFrame(dataframe_filled, columns=dataframe.columns)

    #2. outlier
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit_predict(dataframe)
    df_scores = lof.negative_outlier_factor_
    np.sort(df_scores)[0:40]
    threshold = np.sort(df_scores)[7]
    threshold
    outlier = df_scores > threshold
    dataframe = dataframe[outlier]


    # 7. Robust Scaler
    num_cols.remove("SALARY")

    for col in num_cols:
        transformer = RobustScaler().fit(dataframe[[col]])
        dataframe[col] = transformer.transform(dataframe[[col]])

    return dataframe
df = hitters_data_prep(df)
check_df(df)
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

######################################################
# Base Models
######################################################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


######################################################
# Automated Hyperparameter Optimization
######################################################

#
# cart_params = {'max_depth': range(1, 21),
#                "min_samples_split": range(2, 31)}
#
# rf_params = {"max_depth": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,None],
#              "max_features": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,"auto"],
#              "min_samples_split": [3,5,7,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50],
#              "n_estimators": [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]}
#
# xgboost_params = {"learning_rate": [0.01, 0.1, 0.001,0.0001],
#                   "max_depth": [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50],
#                   "n_estimators": [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
#                   "colsample_bytree": [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]}
#
# lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001,0.0001],
#                    "n_estimators": [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
#                    "colsample_bytree": [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]}
#
# regressors = [("CART", DecisionTreeRegressor(), cart_params),
#               ("RF", RandomForestRegressor(), rf_params),
#               ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
#               ('LightGBM', LGBMRegressor(), lightgbm_params)]
#
# best_models = {}
#
# for name, regressor, params in regressors:
#     print(f"########## {name} ##########")
#     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
#     print(f"RMSE: {round(rmse, 4)} ({name}) ")
#
#     gs_best = GridSearchCV(regressor, params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
#
#     final_model = regressor.set_params(**gs_best.best_params_)
#     rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
#     print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
#
#     print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
#
#     best_models[name] = final_model


######################################################
# Automated Hyperparameter Optimization
######################################################


#cart_params = {'max_depth': range(1, 20),
               #"min_samples_split": range(2, 30)}

#rf_params = {"max_depth": [5,6, 8,9,10, 15, 20,None],
            # "max_features": [3,5, 7, 10,"auto"],
             #"min_samples_split": [5,8, 15, 20,25,50],
            # "n_estimators": [200, 300,400,500, 1000,1500,2000]}

# #xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
#                   "max_depth": [5, 8, 12, 20,25,30,50],
#                   "n_estimators": [100, 200, 300, 500,1000,1500],
#                   "colsample_bytree": [0.1,0.5, 0.7,0.8, 1]}

# #lightgbm_params = {'num_leaves': 3,
#                    'n_estimators': 250,
#                    'min_data_in_leaf': 15,
#                    'min_child_samples': 5,
#                    'max_depth': 8,
#                    'learning_rate': 0.03,
#                    'feature_fraction': 0.2,
#                    'colsample_bytree': 0.1}


#
# cart_params = {'max_depth': range(1, 20),
#                "min_samples_split": range(2, 30)}
#
# rf_params = {"max_depth": [5, 16, None],
#              "max_features": [5, 7, "auto"],
#              "min_samples_split": [2, 15, 20],
#              "n_estimators": [100, 500,600],
#             "min_weight_fraction_leaf" : [0.1,0.0],
#              "min_samples_leaf" : [3,5,1]}
#
# # xgboost_params = {'reg_lambda': [0.5,0.6,0.7,0.8],
# #                   'reg_alpha': [0.01,0.02,0.03],
# #                   "learning_rate": [0.01,0.02,0.03],
# #                   "max_depth": [10,11,12,13,14],
# #                   "n_estimators": [800, 825,840,850],
# #                   'min_child_weight': [0.8,0.9,1],
# #                   "colsample_bytree": [0.3,0.4,0.5]}
#
# lightgbm_params = {"learning_rate": [0.01, 0.1],
#                    "n_estimators": [100, 200, 300, 500],
#                    "colsample_bytree": [0.7, 1]}
#
# regressors = [  ("CART", DecisionTreeRegressor(), cart_params),
#     ("RF", RandomForestRegressor(), rf_params),
#     # ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
#     ('LightGBM', LGBMRegressor(boosting_type='dart'), lightgbm_params)]
#
# best_models = {}
#
# for name, regressor, params in regressors:
#     print(f"########## {name} ##########")
#     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
#     print(f"RMSE: {round(rmse, 4)} ({name}) ")
#
#     gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
#
#     final_model = regressor.set_params(**gs_best.best_params_)
#     rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
#     print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
#
#     print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

#randomcv RF
rf_model = RandomForestRegressor(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 20, 10),
                    "max_features": np.random.randint(2, 50, 20),
                    "min_samples_split": np.random.randint(2, 20, 10),
                    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1500, num=40)],
                    "min_samples_leaf" : np.random.randint(2, 50, 20),
                    "min_weight_fraction_leaf" : [0.01,0.1,0.2,0.3,0.02,0.5],
                    "min_impurity_decrease":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                    "max_samples":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

# En iyi hiperparametre değerleri:
rf_random.best_params_

# En iyi skor
rf_random.best_score_


#randomcv LGBM
lgb_model = LGBMRegressor(random_state=17)

lgb_random_params = {"num_leaves" : np.random.randint(2, 10, 5),
                     "max_depth": np.random.randint(2, 20, 10),
                     "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=50)],
                     "min_child_samples": np.random.randint(5, 20, 10),
                     "reg_alpha": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_lambda": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9,1,3,5,7],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.001,0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}


lgb_random = RandomizedSearchCV(estimator=lgb_model,param_distributions=lgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)


lgb_random.fit(X, y)

# En iyi hiperparametre değerleri:
lgb_random.best_params_

# En iyi skor
lgb_random.best_score_


#XGB Random Search
xgb_model = XGBRegressor(random_state=17)


xgb_random_params = {"max_depth": np.random.randint(2, 20, 20),
                     "n_estimators": [int(x) for x in np.linspace(start=100, stop=2000, num=50)],
                     "gamma": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "min_child_weight": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_alpha": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_lambda": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.001,0.01,0.1,0.2,0.3,0.02,0.5,]}


xgb_random = RandomizedSearchCV(estimator=xgb_model,param_distributions=xgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1,)


xgb_random.fit(X, y)


# En iyi hiperparametre değerleri:
xgb_random.best_params_

# En iyi skor
xgb_random.best_score_


#testing
regressors = [#("CART", DecisionTreeRegressor(), cart_params),
    ("RF", RandomForestRegressor(),rf_random.best_params_),
    ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgb_random.best_params_),
    ('LightGBM', LGBMRegressor(), lgb_random.best_params_)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    #gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**params)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {lgb_random.best_params_}", end="\n\n")

    best_models[name] = final_model


######################################################
# # Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

######################################################
# Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)




















