# subject independent testing
# 按不同受试者重新构建训练集和测试集
from functions import dataread_inter, evaluation_metrics, bland_altman_plot, BHS_compute
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # load data
    PATH_SBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\20211021\SBP"
    PATH_DBP = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\20211021\DBP"

    subject_list = list(range(13))
    subject_file_list = [r"\20211021", r"\20211022_h", r"\20211022_l", r"\20211024",
                         r"\20211026", r"\20211027", r"\20211028", r"\20211028c",
                         r"\20211029", r"\20211102", r"\20211103", r"\20211105",
                        r"\20211116"]

    random.seed(21)
    index = random.sample(range(0, 13), 3)
    subject_file_list_test = [subject_file_list[index[0]], subject_file_list[index[1]], \
                             subject_file_list[11], subject_file_list[index[2]]]
    subject_file_list_train = []
    for ii_file in subject_file_list:
        if ii_file not in subject_file_list_test:
            subject_file_list_train.append(ii_file)

    PATH_base = r"F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new"
    PATH_sbp_tail, PATH_dbp_tail = "\SBP", "\DBP"

    x_train_sbp, y_train_sbp = np.zeros((1, 48)), np.zeros(1)
    x_train_dbp, y_train_dbp = np.zeros((1, 48)), np.zeros(1)
    # Training set construct
    for subject_file in subject_file_list_train:
        PATH_SBP = PATH_base + subject_file + PATH_sbp_tail
        PATH_DBP = PATH_base + subject_file + PATH_dbp_tail
        # Load data
        X_sbp_subject, y_sbp_subject = dataread_inter(PATH_SBP)
        X_dbp_subject, y_dbp_subject = dataread_inter(PATH_DBP)

        x_train_sbp = np.vstack((x_train_sbp, X_sbp_subject))
        y_train_sbp = np.vstack((y_train_sbp, y_sbp_subject))
        x_train_dbp = np.vstack((x_train_dbp, X_dbp_subject))
        y_train_dbp = np.vstack((y_train_dbp, y_dbp_subject))

    x_train_sbp = np.delete(x_train_sbp, 0, axis=0)
    y_train_sbp = np.delete(y_train_sbp, 0, axis=0)
    x_train_dbp = np.delete(x_train_dbp, 0, axis=0)
    y_train_dbp = np.delete(y_train_dbp, 0, axis=0)

    # Testing set construct
    x_test_sbp, y_test_sbp = np.zeros((1, 48)), np.zeros(1)
    x_test_dbp, y_test_dbp = np.zeros((1, 48)), np.zeros(1)
    for subject_file in subject_file_list_test:
        PATH_SBP = PATH_base + subject_file + PATH_sbp_tail
        PATH_DBP = PATH_base + subject_file + PATH_dbp_tail
        # Load data
        X_sbp_subject, y_sbp_subject = dataread_inter(PATH_SBP)
        X_dbp_subject, y_dbp_subject = dataread_inter(PATH_DBP)

        x_test_sbp = np.vstack((x_test_sbp, X_sbp_subject))
        y_test_sbp = np.vstack((y_test_sbp, y_sbp_subject))
        x_test_dbp = np.vstack((x_test_dbp, X_dbp_subject))
        y_test_dbp = np.vstack((y_test_dbp, y_dbp_subject))

    x_test_sbp = np.delete(x_test_sbp, 0, axis=0)
    y_test_sbp = np.delete(y_test_sbp, 0, axis=0)
    x_test_dbp = np.delete(x_test_dbp, 0, axis=0)
    y_test_dbp = np.delete(y_test_dbp, 0, axis=0)


    # idx = np.shape(x_train_sbp)[1] - 5  # 不包含受试者身份特征的起始下标
    # x_train_sbp = x_train_sbp[:, 0:idx]
    # x_test_sbp = x_test_sbp[:, 0:idx]
    # x_train_dbp = x_train_dbp[:, 0:idx]
    # x_test_dbp = x_test_dbp[:, 0:idx]

    # # 按照特征排序筛选的特征
    # idx_ranking = [14, 42, 0, 15, 2, 4, 38, 32, 33, 41, 3, 5, 27, 34, 16, 6, 25, 1, 28, 23, 35, 30, 9, 26, 37, 40, 17,
    #                31, 7, 10, 24, 11, 18, 12, 22, 8, 36, 21, 13, 29, 39, 20, 19, ]
    # K = 25
    # idx_select = idx_ranking[0:K]  # 挑选Top-K个特征
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

    # BHS result
    CP_5_sbp, CP_10_sbp, CP_15_sbp = BHS_compute(y_pred_sbp_rf, y_test_sbp)
    print(CP_5_sbp, CP_10_sbp, CP_15_sbp)
    CP_5_dbp, CP_10_dbp, CP_15_dbp = BHS_compute(y_pred_dbp_rf, y_test_dbp)
    print(CP_5_dbp, CP_10_dbp, CP_15_dbp)

    # Draw pictures
    y_pred_sbp = y_pred_sbp_rf
    y_pred_dbp = y_pred_dbp_rf

    # correlation plot
    feature_name = ["PTT_max","PTT_min","PAT","CT",
                    "DW","DW_25","DW_50","DW_75","DW_90",
                    "SW","SW_25","SW_50","SW_75","SW_90",
                    "PW","PW_25","PW_50","PW_75","PW_90",
                    "PWR_25","PWR_50","PWR_75","PWR_90",
                    "AM_max","AM_min","AM_MD","PP",
                    "AR_max","AR_MD","AS","DS",
                    "AMd_max","PWd","PWd_50","PWRd","ASd","DSd",
                    "SD","Skew","Kurt","ApEn","SampEn",'HR',
                    'height','weight','BMI','age','gender']
    feature_name = list(np.array(feature_name))
    dataset_train_sbp = pd.DataFrame(np.hstack((x_test_sbp, y_pred_sbp, y_test_sbp)), columns=feature_name+['Reference SBP (mmHg)','Estimated SBP (mmHg)'])
    dataset_train_dbp = pd.DataFrame(np.hstack((x_test_dbp, y_pred_dbp, y_test_dbp)), columns=feature_name+['Reference DBP (mmHg)','Estimated DBP (mmHg)'])
    sns.set_theme(style='white', font_scale=2)
    # sns.set(font_scale=1.7)
    plt.figure(1)
    sns.regplot(x='Reference SBP (mmHg)', y='Estimated SBP (mmHg)', data=dataset_train_sbp)
    plt.figure(2)
    sns.regplot(x='Reference DBP (mmHg)', y='Estimated DBP (mmHg)', data=dataset_train_dbp)

    # bland_altman plot
    plt.figure(3)
    bland_altman_plot(y_pred_sbp_rf, y_test_sbp, X_axis='Mean of SBP (mmHg)')
    plt.figure(4)
    bland_altman_plot(y_pred_dbp_rf, y_test_dbp, X_axis='Mean of DBP (mmHg)')

    # # histogram
    # dataset_train_sbp['Blood Pressure'] = ['SBP']*len(dataset_train_sbp)
    # dataset_train_dbp['Blood Pressure'] = ['DBP']*len(dataset_train_dbp)
    # dataset_df_all = pd.concat([dataset_train_sbp, dataset_train_dbp], axis=0)
    # dataset_df_all = dataset_df_all.reset_index(drop=True)
    #
    # # sns.displot(dataset_train_sbp, x="Reference BP (mmHg)", binwidth=5)
    # sns.displot(dataset_df_all, x="Reference BP (mmHg)", binwidth=3.8, hue='Blood Pressure')
    # plt.show()