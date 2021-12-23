# subject dependent testing
# 7:3训练和测试（训练集和测试集已经划分好）
from functions import dataread, evaluation_metrics, bland_altman_plot, BHS_compute, evaluation_metrics_old
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    # load data
    PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\SBP"
    PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\DBP"

    x_train_sbp, x_test_sbp, y_train_sbp, y_test_sbp = dataread(PATH_SBP)
    x_train_dbp, x_test_dbp, y_train_dbp, y_test_dbp = dataread(PATH_DBP)

    idx = np.shape(x_train_sbp)[1] - 5  # 不包含受试者身份特征的起始下标
    x_train_sbp = x_train_sbp[:, 0:idx]
    x_test_sbp = x_test_sbp[:, 0:idx]
    x_train_dbp = x_train_dbp[:, 0:idx]
    x_test_dbp = x_test_dbp[:, 0:idx]

    # 按照特征排序筛选的特征
    idx_ranking = [14,42,0,15,2,4,38,32,33,41,3,5,27,34,16,6,25,1,28,23,35,30,9,26,37,40,17,31,7,10,24,11,18,12,22,8,36,21,13,29,39,20,19,]
    K = 25
    idx_select = idx_ranking[0:K] # 挑选Top-K个特征
    x_train_sbp = x_train_sbp[:, idx_select]
    x_test_sbp = x_test_sbp[:, idx_select]
    x_train_dbp = x_train_dbp[:, idx_select]
    x_test_dbp = x_test_dbp[:, idx_select]

    # # 除去利用ECG的特征(PTT_MAX,PTT_MIN,PAT,HR)
    # idx_select = [0,3] + list(range(5,17)) + list(range(18,25))     # 挑选不使用ECG的特征
    # x_train_sbp = x_train_sbp[:, idx_select]
    # x_test_sbp = x_test_sbp[:, idx_select]
    # x_train_dbp = x_train_dbp[:, idx_select]
    # x_test_dbp = x_test_dbp[:, idx_select]

    # model definition
    model_sbp = linear_model.LinearRegression()
    model_sbp.fit(x_train_sbp, y_train_sbp)
    model_dbp = linear_model.LinearRegression()
    model_dbp.fit(x_train_dbp, y_train_dbp)

    # Random Forest
    model_sbp_rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=0)
    model_dbp_rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=0)
    # model_sbp_rf = DecisionTreeRegressor(random_state=0)
    # model_dbp_rf = DecisionTreeRegressor(random_state=0)
    # model_sbp_rf = SVR(C=1000)
    # model_dbp_rf = SVR(C=1000)

    model_sbp_rf.fit(x_train_sbp, y_train_sbp)
    model_dbp_rf.fit(x_train_dbp, y_train_dbp)

    # predict
    # LR
    y_pred_sbp = model_sbp.predict(x_test_sbp)
    y_pred_dbp = model_dbp.predict(x_test_dbp)

    # RF
    y_pred_sbp_rf = model_sbp_rf.predict(x_test_sbp)
    y_pred_dbp_rf = model_dbp_rf.predict(x_test_dbp)
    y_pred_sbp_rf = np.expand_dims(y_pred_sbp_rf, axis=1)
    y_pred_dbp_rf = np.expand_dims(y_pred_dbp_rf, axis=1)

    # evaluation metrics
    ME_sbp, RMSE_sbp, MAE_sbp, R_sbp = evaluation_metrics(y_pred_sbp, y_test_sbp)
    ME_dbp, RMSE_dbp, MAE_dbp, R_dbp = evaluation_metrics(y_pred_dbp, y_test_dbp)
    print("SBP:")
    print(ME_sbp, RMSE_sbp, MAE_sbp, R_sbp)
    print("DBP:")
    print(ME_dbp, RMSE_dbp, MAE_dbp, R_dbp)

    ME_sbp_rf, RMSE_sbp_rf, MAE_sbp_rf, R_sbp_rf = evaluation_metrics(y_pred_sbp_rf, y_test_sbp)
    ME_dbp_rf, RMSE_dbp_rf, MAE_dbp_rf, R_dbp_rf = evaluation_metrics(y_pred_dbp_rf, y_test_dbp)
    print("SBP:")
    print(ME_sbp_rf, RMSE_sbp_rf, MAE_sbp_rf, R_sbp_rf)
    print("DBP:")
    print(ME_dbp_rf, RMSE_dbp_rf, MAE_dbp_rf, R_dbp_rf)


    # BHS result
    CP_5_sbp, CP_10_sbp, CP_15_sbp = BHS_compute(y_pred_sbp_rf, y_test_sbp)
    print(CP_5_sbp, CP_10_sbp, CP_15_sbp)
    CP_5_dbp, CP_10_dbp, CP_15_dbp = BHS_compute(y_pred_dbp_rf, y_test_dbp)
    print(CP_5_dbp, CP_10_dbp, CP_15_dbp)


    # histogram
    error_sbp = y_pred_sbp_rf - y_test_sbp
    error_dbp = y_pred_dbp_rf - y_test_dbp
    dataset_error = pd.DataFrame(np.hstack((error_sbp, error_dbp)), columns=['SBP estimation error (mmHg)','DBP estimation error (mmHg)'])
    sns.set_theme(style='white', font_scale=1.8)
    sns.displot(dataset_error, x="SBP estimation error (mmHg)")
    sns.displot(dataset_error, x="DBP estimation error (mmHg)")



    # # Draw pictures
    # y_pred_sbp = y_pred_sbp_rf
    # y_pred_dbp = y_pred_dbp_rf

    # # correlation plot
    # feature_name = ["PTT_max","PTT_min","PAT","CT",
    #                 "DW","DW_25","DW_50","DW_75","DW_90",
    #                 "SW","SW_25","SW_50","SW_75","SW_90",
    #                 "PW","PW_25","PW_50","PW_75","PW_90",
    #                 "PWR_25","PWR_50","PWR_75","PWR_90",
    #                 "AM_max","AM_min","AM_MD","PP",
    #                 "AR_max","AR_MD","AS","DS",
    #                 "AMd_max","PWd","PWd_50","PWRd","ASd","DSd",
    #                 "SD","Skew","Kurt","ApEn","SampEn",'HR']
    # feature_name = list(np.array(feature_name)[idx_select])
    # dataset_train_sbp = pd.DataFrame(np.hstack((x_test_sbp, y_pred_sbp, y_test_sbp)), columns=feature_name+['Reference SBP (mmHg)','Estimated SBP (mmHg)'])
    # dataset_train_dbp = pd.DataFrame(np.hstack((x_test_dbp, y_pred_dbp, y_test_dbp)), columns=feature_name+['Reference DBP (mmHg)','Estimated DBP (mmHg)'])
    # sns.set_theme(style='white', font_scale=2)
    # # sns.set(font_scale=1.7)
    # plt.figure(1)
    # sns.regplot(x='Reference SBP (mmHg)', y='Estimated SBP (mmHg)', data=dataset_train_sbp)
    # plt.figure(2)
    # sns.regplot(x='Reference DBP (mmHg)', y='Estimated DBP (mmHg)', data=dataset_train_dbp)
    #
    # # bland_altman plot
    # plt.figure(3)
    # bland_altman_plot(y_pred_sbp_rf, y_test_sbp, X_axis='Mean of SBP (mmHg)')
    # plt.figure(4)
    # bland_altman_plot(y_pred_dbp_rf, y_test_dbp, X_axis='Mean of DBP (mmHg)')


    # # histogram
    # x_train_sbp = np.vstack((x_train_sbp, x_test_sbp))
    # x_train_dbp = np.vstack((x_train_dbp, x_test_dbp))
    # y_train_sbp = np.vstack((y_train_sbp, y_test_sbp))
    # y_train_dbp = np.vstack((y_train_dbp, y_test_dbp))
    # feature_name = ["PTT_max", "PTT_min", "PAT", "CT",
    #                 "DW", "DW_25", "DW_50", "DW_75", "DW_90",
    #                 "SW", "SW_25", "SW_50", "SW_75", "SW_90",
    #                 "PW", "PW_25", "PW_50", "PW_75", "PW_90",
    #                 "PWR_25", "PWR_50", "PWR_75", "PWR_90",
    #                 "AM_max", "AM_min", "AM_MD", "PP",
    #                 "AR_max", "AR_MD", "AS", "DS",
    #                 "AMd_max", "PWd", "PWd_50", "PWRd", "ASd", "DSd",
    #                 "SD", "Skew", "Kurt", "ApEn", "SampEn", 'HR']
    # feature_name = list(np.array(feature_name)[idx_select])
    # dataset_train_sbp = pd.DataFrame(np.hstack((x_train_sbp, y_train_sbp)),
    #                                  columns=feature_name + ['Reference BP (mmHg)'])
    # dataset_train_dbp = pd.DataFrame(np.hstack((x_train_dbp, y_train_dbp)),
    #                                  columns=feature_name + ['Reference BP (mmHg)'])
    #
    # sns.set_theme(style='white', font_scale=2)
    # dataset_train_sbp['Blood Pressure'] = ['SBP']*len(dataset_train_sbp)
    # dataset_train_dbp['Blood Pressure'] = ['DBP']*len(dataset_train_dbp)
    # dataset_df_all = pd.concat([dataset_train_sbp, dataset_train_dbp], axis=0)
    # dataset_df_all = dataset_df_all.reset_index(drop=True)
    #
    # # sns.displot(dataset_train_sbp, x="Reference BP (mmHg)", binwidth=5)
    # sns.displot(dataset_df_all, x="Reference BP (mmHg)", binwidth=7, hue='Blood Pressure')
    # plt.show()
