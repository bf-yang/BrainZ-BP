% 特征提取函数，从数据库中读入每个ECG和脑阻抗片段
% 提取43维特征，存成.mat文件
% 存特征和血压，在python中进行回归预测
% 对每个受试者特征提取
% 不预先划分训练和测试
% 将所有特征X和标签y存到对应受试者的文件夹
clc; clear all;
fs = 500;
% 按自然顺序读取文件

file_name = '20211116';
path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\';
path = [path, file_name];

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

% 依次读取每个样本signal
nsamples = length(dir_cell); 
fea_ptt_max = [];
fea_ptt_min = [];
fea_ptt_dif = [];
fea_hr = [];
SBP = [];
DBP = [];
H = [];
W = [];
A = [];
G = [];
t_segment = 8; % 时间窗长度
features = [];
for i = 1: nsamples
    disp(i);
% for i = 113
    % 读取每个样本的ecg和reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    % 按格式读取信号和标签
    ecg = signal(1,1:4000);  % ECG
    reg = signal(2,1:4000);  % REG
    sbp = signal(1,4001);     
    dbp = signal(1,4002);
    hr = signal(1,4003);
    age = signal(1,end);
    height = signal(2,4001);  
    weight = signal(2,4002);  
    r_wave_para = signal(2,4003);  % 心电R波检测参数
    gender = signal(2,4004);
    bmi = (weight/2)/(height/100)^2;
    
    % 提取特征
    [features_segment, R_index, Imped_index, Imped_index2, Imped_index3] = feature_extract(ecg, reg, fs, r_wave_para);
    features_segment = [features_segment, hr, height, weight, bmi, age, gender, sbp, dbp];
%     features_segment = [features_segment, height, weight, bmi, age, gender, sbp, dbp];

    features = [features; features_segment];
end

% 划分训练集和测试集
array = [features(:,1:end-2), features(:,end-1)];
% array = [features(:,1:end-2), features(:,end)];

% X:特征，y：标签
X = array(:,1:end-1);
y = array(:,end);

% % 存特征
% path_1 = 'F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\';
% path_2 = '\SBP\';
% path_f = [path_1,file_name,path_2];
% save(fullfile(path_f, 'X'),'X','-v7.3');
% % 存标签
% path_1 = 'F:\Project-342B\血压预测\Code\blood_pressure_py\Data\Database_new\';
% path_2 = '\SBP\';
% path_l = [path_1,file_name,path_2];
% save(fullfile(path_l, 'y'),'y','-v7.3');


