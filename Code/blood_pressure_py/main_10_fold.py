# 10-fold cross validation
# subject-dependent testing
from functions import dataread, evaluation_metrics, bland_altman_plot
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold

# Intra-patient experiment
# 10-fold cross validation
if __name__ == '__main__':
    # load data
    PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\SBP"
    PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\DBP"

    x_train_sbp, x_test_sbp, y_train_sbp, y_test_sbp = dataread(PATH_SBP)
    x_train_dbp, x_test_dbp, y_train_dbp, y_test_dbp = dataread(PATH_DBP)
    # 先拼起来用于10-fold交叉验证
    X_sbp = np.vstack((x_train_sbp, x_test_sbp))
    y_sbp = np.vstack((y_train_sbp, y_test_sbp))
    X_dbp = np.vstack((x_train_dbp, x_test_dbp))
    y_dbp = np.vstack((y_train_dbp, y_test_dbp))

    # 不使用受试者身份特征（身高，体重，BMI）
    idx = np.shape(x_train_sbp)[1] - 5  # 不包含受试者身份特征的起始下标
    X_sbp = X_sbp[:, 0:idx]
    X_dbp = X_dbp[:, 0:idx]

    # 按照特征排序筛选的特征
    idx_ranking = [14, 42, 0, 15, 2, 4, 38, 32, 33, 41, 3, 5, 27, 34, 16, 6, 25, 1, 28, 23, 35, 30, 9, 26, 37, 40, 17,
                   31, 7, 10, 24, 11, 18, 12, 22, 8, 36, 21, 13, 29, 39, 20, 19, ]
    K = 25
    idx_select = idx_ranking[0:K]  # 挑选Top-K个特征
    X_sbp = X_sbp[:, idx_select]
    X_dbp = X_dbp[:, idx_select]

    # # 除去利用ECG的特征(PTT_MAX,PTT_MIN,PAT,HR)
    # idx_select = [0,3] + list(range(5,17)) + list(range(18,25))     # 挑选不使用ECG的特征
    # X_sbp = X_sbp[:, idx_select]
    # X_dbp = X_dbp[:, idx_select]


    # K-fold cross validation
    KF = KFold(n_splits=10, shuffle=True, random_state=1)
    res_sbp = []
    res_dbp = []
    for train_idx, test_idx in KF.split(X_sbp):
        # 每fold用于训练和测试的数据
        x_train_sbp_kf, x_test_sbp_kf = X_sbp[train_idx], X_sbp[test_idx]
        x_train_dbp_kf, x_test_dbp_kf = X_dbp[train_idx], X_dbp[test_idx]
        y_train_sbp_kf, y_test_sbp_kf = y_sbp[train_idx], y_sbp[test_idx]
        y_train_dbp_kf, y_test_dbp_kf = y_dbp[train_idx], y_dbp[test_idx]

        # model definition
        # Random Forest
        model_sbp_rf = RandomForestRegressor(n_estimators=500, max_depth=100, min_samples_leaf=1, random_state=1)
        model_dbp_rf = RandomForestRegressor(n_estimators=500, max_depth=100, min_samples_leaf=1, random_state=1)
        # model_sbp_rf = DecisionTreeRegressor(random_state=1)
        # model_dbp_rf = DecisionTreeRegressor(random_state=1)
        # model_sbp_rf = SVR(C=1000)
        # model_dbp_rf = SVR(C=1000)
        # model_sbp_rf = linear_model.LinearRegression()
        # model_dbp_rf = linear_model.LinearRegression()

        model_sbp_rf.fit(x_train_sbp_kf, y_train_sbp_kf)
        model_dbp_rf.fit(x_train_dbp_kf, y_train_dbp_kf)

        # predict
        # RF
        y_pred_sbp_rf = model_sbp_rf.predict(x_test_sbp_kf)
        y_pred_dbp_rf = model_dbp_rf.predict(x_test_dbp_kf)
        y_pred_sbp_rf = np.expand_dims(y_pred_sbp_rf, axis=1)
        y_pred_dbp_rf = np.expand_dims(y_pred_dbp_rf, axis=1)

        # evaluation metrics
        ME_sbp_rf, RMSE_sbp_rf, MAE_sbp_rf, R_sbp_rf = evaluation_metrics(y_pred_sbp_rf, y_test_sbp_kf)
        ME_dbp_rf, RMSE_dbp_rf, MAE_dbp_rf, R_dbp_rf = evaluation_metrics(y_pred_dbp_rf, y_test_dbp_kf)
        print("SBP:")
        print(ME_sbp_rf, RMSE_sbp_rf, MAE_sbp_rf, R_sbp_rf)
        print("DBP:")
        print(ME_dbp_rf, RMSE_dbp_rf, MAE_dbp_rf, R_dbp_rf)


        res_sbp.append([ME_sbp_rf, RMSE_sbp_rf, MAE_sbp_rf, R_sbp_rf])
        res_dbp.append([ME_dbp_rf, RMSE_dbp_rf, MAE_dbp_rf, R_dbp_rf])

    res_sbp_arr, res_dbp_arr = np.array(res_sbp), np.array(res_dbp)
    # 10-fold cross validation 均值：
    Mean_sbp = res_sbp_arr.mean(axis=0)
    Mean_dbp = res_dbp_arr.mean(axis=0)
    print(Mean_sbp)
    print(Mean_dbp)
    # 10-fold cross validation 标准差：
    Std_sbp = res_sbp_arr.std(axis=0)
    Std_dbp = res_dbp_arr.std(axis=0)
    print(Std_sbp)
    print(Std_dbp)



