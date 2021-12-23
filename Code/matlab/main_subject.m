% ������ȡ�����������ݿ��ж���ÿ��ECG�����迹Ƭ��
% ��ȡ43ά���������.mat�ļ�
% ��������Ѫѹ����python�н��лع�Ԥ��
% ��ÿ��������������ȡ
% ��Ԥ�Ȼ���ѵ���Ͳ���
% ����������X�ͱ�ǩy�浽��Ӧ�����ߵ��ļ���
clc; clear all;
fs = 500;
% ����Ȼ˳���ȡ�ļ�

file_name = '20211116';
path = 'F:\Project-342B\ѪѹԤ��\BloodPressure_Prediction\Dataset_seg\';
path = [path, file_name];

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

% ���ζ�ȡÿ������signal
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
t_segment = 8; % ʱ�䴰����
features = [];
for i = 1: nsamples
    disp(i);
% for i = 113
    % ��ȡÿ��������ecg��reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    % ����ʽ��ȡ�źźͱ�ǩ
    ecg = signal(1,1:4000);  % ECG
    reg = signal(2,1:4000);  % REG
    sbp = signal(1,4001);     
    dbp = signal(1,4002);
    hr = signal(1,4003);
    age = signal(1,end);
    height = signal(2,4001);  
    weight = signal(2,4002);  
    r_wave_para = signal(2,4003);  % �ĵ�R��������
    gender = signal(2,4004);
    bmi = (weight/2)/(height/100)^2;
    
    % ��ȡ����
    [features_segment, R_index, Imped_index, Imped_index2, Imped_index3] = feature_extract(ecg, reg, fs, r_wave_para);
    features_segment = [features_segment, hr, height, weight, bmi, age, gender, sbp, dbp];
%     features_segment = [features_segment, height, weight, bmi, age, gender, sbp, dbp];

    features = [features; features_segment];
end

% ����ѵ�����Ͳ��Լ�
array = [features(:,1:end-2), features(:,end-1)];
% array = [features(:,1:end-2), features(:,end)];

% X:������y����ǩ
X = array(:,1:end-1);
y = array(:,end);

% % ������
% path_1 = 'F:\Project-342B\ѪѹԤ��\Code\blood_pressure_py\Data\Database_new\';
% path_2 = '\SBP\';
% path_f = [path_1,file_name,path_2];
% save(fullfile(path_f, 'X'),'X','-v7.3');
% % ���ǩ
% path_1 = 'F:\Project-342B\ѪѹԤ��\Code\blood_pressure_py\Data\Database_new\';
% path_2 = '\SBP\';
% path_l = [path_1,file_name,path_2];
% save(fullfile(path_l, 'y'),'y','-v7.3');


