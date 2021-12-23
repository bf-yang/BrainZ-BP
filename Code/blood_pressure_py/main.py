# subject-dependent testing
# 绘制特征重要性并存储
# 1. 皮尔逊相关系数
# 2. 随机森林杂质

from functions import dataread, evaluation_metrics, bland_altman_plot
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
import pandas as pd
import graphviz

if __name__ == '__main__':
    # load data
    # PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\SBP"
    # PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\DBP"

    PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\SBP"
    PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\total\DBP"
    # PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database\20211026\SBP"
    # PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database\20211026\DBP"

    # PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_HRcomput\total\SBP"
    # PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_HRcomput\total\DBP"
    # PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_HRcomput\20211026\SBP"
    # PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_HRcomput\20211026\DBP"

    x_train_sbp, x_test_sbp, y_train_sbp, y_test_sbp = dataread(PATH_SBP)
    x_train_dbp, x_test_dbp, y_train_dbp, y_test_dbp = dataread(PATH_DBP)

    idx = np.shape(x_train_sbp)[1] - 5  # 不包含受试者身份特征的起始下标
    x_train_sbp = x_train_sbp[:, 0:idx]
    x_test_sbp = x_test_sbp[:, 0:idx]
    x_train_dbp = x_train_dbp[:, 0:idx]
    x_test_dbp = x_test_dbp[:, 0:idx]

    # # 按照特征排序筛选的特征
    # idx_ranking = [14,42,0,15,2,4,38,32,33,41,3,5,27,34,16,6,25,1,28,23,35,30,9,26,37,40,17,31,7,10,24,11,18,12,22,8,36,21,13,29,39,20,19,]
    # idx_select = [14,42,0,15,2,4,38,32,33,41,3,5,27,34,16,6,25,1,28,23]
    # # idx_select = list(range(3, 42))
    # x_train_sbp = x_train_sbp[:, idx_select]
    # x_test_sbp = x_test_sbp[:, idx_select]
    # x_train_dbp = x_train_dbp[:, idx_select]
    # x_test_dbp = x_test_dbp[:, idx_select]

    # # feature normalization
    # scaler = StandardScaler().fit(x_train_sbp)
    # x_train_sbp = scaler.transform(x_train_sbp)
    # x_test_sbp = scaler.transform(x_test_sbp)

    # model definition
    model_sbp = linear_model.LinearRegression()
    model_sbp.fit(x_train_sbp, y_train_sbp)
    model_dbp = linear_model.LinearRegression()
    model_dbp.fit(x_train_dbp, y_train_dbp)

    # Random Forest
    model_sbp_rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=1)
    model_dbp_rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=1)
    # model_sbp_rf = DecisionTreeRegressor(random_state=1)
    # model_dbp_rf = DecisionTreeRegressor(random_state=1)
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
    err_mean_sbp, err_std_sbp, err_mean_abs_sbp = evaluation_metrics(y_pred_sbp, y_test_sbp)
    err_mean_dbp, err_std_dbp, err_mean_abs_dbp = evaluation_metrics(y_pred_dbp, y_test_dbp)
    R_sbp = r2_score(y_test_sbp, y_pred_sbp)
    R_dbp = r2_score(y_test_dbp, y_pred_dbp)
    print("SBP:")
    print(err_mean_sbp, err_std_sbp, err_mean_abs_sbp, R_sbp)
    print("DBP:")
    print(err_mean_dbp, err_std_dbp, err_mean_abs_dbp, R_dbp)

    err_mean_sbp_rf, err_std_sbp_rf, err_mean_abs_sbp_rf = evaluation_metrics(y_pred_sbp_rf, y_test_sbp)
    R_sbp_rf = r2_score(y_test_sbp, y_pred_sbp_rf)
    err_mean_dbp_rf, err_std_dbp_rf, err_mean_abs_dbp_rf = evaluation_metrics(y_pred_dbp_rf, y_test_dbp)
    R_dbp_rf = r2_score(y_test_dbp, y_pred_dbp_rf)
    print("RF SBP:")
    print(err_mean_sbp_rf, err_std_sbp_rf, err_mean_abs_sbp_rf, R_sbp_rf)
    print("RF DBP:")
    print(err_mean_dbp_rf, err_std_dbp_rf, err_mean_abs_dbp_rf, R_dbp_rf)

    importances_values = model_sbp_rf.feature_importances_
    importances = pd.DataFrame(importances_values, columns=["importance"])
    # importances.to_csv('feature_importance_RF.csv')
    # # 决策树可视化
    # fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    # tree.plot_tree(model_sbp_rf)
    # fig.savefig('Decision Tree.png')

    # Feature importance
    res_sbp = []
    for ii in range(np.shape(x_train_sbp)[1]):
        x = x_train_sbp[:, ii]
        y = np.squeeze(y_train_sbp)
        res_sbp.append(pearsonr(x, y))  # PCC
    res_sbp = np.array(res_sbp)
    # # Save feature importance R
    data = pd.DataFrame(res_sbp)
    # data.to_csv('feature_importance_R.csv')



    # # estimator = SVR(kernel="linear")
    # # estimator = linear_model.LinearRegression()
    # estimator = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=1)
    # selector = RFE(estimator, n_features_to_select=5, step=1)
    # selector = selector.fit(x_train_sbp, y_train_sbp)
    # res = selector.support_
    # rank = selector.ranking_



    # y_pred_sbp = y_pred_sbp_rf
    # y_pred_dbp = y_pred_dbp_rf
    #
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
    # dataset_train_sbp = pd.DataFrame(np.hstack((x_test_sbp, y_pred_sbp, y_test_sbp)), columns=feature_name+['Reference BP (mmHg)','Estimated BP (mmHg)'])
    # dataset_train_dbp = pd.DataFrame(np.hstack((x_test_dbp, y_pred_dbp, y_test_dbp)), columns=feature_name+['Reference BP (mmHg)','Estimated BP (mmHg)'])
    # plt.figure(1)
    # sns.regplot(x='Reference BP (mmHg)', y='Estimated BP (mmHg)', data=dataset_train_sbp)
    # plt.grid()
    # plt.figure(2)
    # sns.regplot(x='Reference BP (mmHg)', y='Estimated BP (mmHg)', data=dataset_train_dbp)
    # plt.grid()

    # # bland_altman plot
    # plt.figure(3)
    # bland_altman_plot(y_pred_sbp, y_test_sbp)
    # plt.figure(4)
    # bland_altman_plot(y_pred_dbp, y_test_dbp)
    #
    #
    # # histogram
    # dataset_train_sbp['Blood Pressure'] = ['SBP']*len(dataset_train_sbp)
    # dataset_train_dbp['Blood Pressure'] = ['DBP']*len(dataset_train_dbp)
    # dataset_df_all = pd.concat([dataset_train_sbp, dataset_train_dbp], axis=0)
    # dataset_df_all = dataset_df_all.reset_index(drop=True)
    #
    # # sns.displot(dataset_train_sbp, x="Reference BP (mmHg)", binwidth=5)
    # sns.displot(dataset_df_all, x="Reference BP (mmHg)", binwidth=3.8, hue='Blood Pressure')
    # plt.show()
