% 用于将约40秒的.mat信号切分为8s片段，overlapping6s，并存入数据库
% 人眼选择质量好的片段存储
% 存储eeg+reg；SBP,DBP,HR,R波检测参数

% 每个人，每条数据查看
% 将每条数据分割为多段，画出所有片段的图，然后选择指定的idx片段进行存储
% 存储格式：
% signal：第一行是心电，第二行是脑阻抗
% 第一行最后3位分别是SBP，DBP，HR
% 第二行最后3位分别是身高H，体重W，R波检测参数

% 修改 path 改变受试者
% 修改 i 改变查看第几次测量的数据
% 修改 idx_save_list 改变要存储的片段的下标
% 存储时，文件名自动按顺序递增

clc; clear all;
fs = 500;
% 按自然顺序读取文件
% path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\dataset_1';
% path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\data_mix';
path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\data_11.16';
age = 22;

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

% 读取血压值
file_bp = xlsread(fullfile(path, 'BloodPressure.xlsx'));
sbp = file_bp(:,1);
dbp = file_bp(:,2);
hr = file_bp(:,3);
height = file_bp(:,4);
weight = file_bp(:,5);

% 依次读取每个样本signal
% for i = 1: length(dir_cell)
dim = 2;
nsamples = length(dir_cell); 
fea_ptt_max = [];
fea_ptt_min = [];
fea_hr = [];
SBP = [];
DBP = [];
H = [];
W = [];
ECG = [];
REG = [];
t_segment = 8;
overlapping = 6;
for i = 9
    % 读取每个样本的ecg和reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    ecg = signal(1,:);
    reg = signal(2,:);
    
    % N_seg = floor(length(ecg)/fs/t_segment);  % No overlapping
    T = length(ecg)/fs;  % 信号总时间
    move = t_segment - overlapping;
    N_seg = floor((T - t_segment)/(t_segment - overlapping)) + 1;
    % 分段
    idx_start = 1;
    for ii = [1:N_seg]

        % No overlapping
        % idx_start = (ii-1) * t_segment * fs + 1;
        % idx_end = ii * t_segment * fs; 
        % Overlapping
        idx_start = (ii-1) * move * fs + 1;
        idx_end = idx_start + t_segment * fs - 1; 
        
        
        ecg_seg = ecg(idx_start:idx_end);
        reg_seg = reg(idx_start:idx_end);
        
        % 提取特征
        PTT = fea_PTT(ecg_seg, reg_seg, fs);
        fea_ptt_max = [fea_ptt_max; PTT(1)];
        fea_ptt_min = [fea_ptt_min; PTT(2)];
        fea_hr = [fea_hr; hr(i)];
        SBP = [SBP; sbp(i)];
        DBP = [DBP; dbp(i)];
        H = [H, height(i)];
        W = [W, weight(i)];
        ECG = [ECG;ecg_seg];
        REG = [REG;reg_seg];
    end
end

% 看波形
r_wave_para = 250;
for view_num = [1:N_seg]  % 查看一次测量分割为N_seg个片段的波形
    % view_num = 2;
    ecg = ECG(view_num,:);
    reg = REG(view_num,:);

    [PTT, R_index, Imped_index, Imped_index2, Imped_index3] = fea_PTT_view(ecg, reg, fs, r_wave_para);

    % 绘图（ecg与ekg在两张图上，查看R波和极小值的检测效果）
    figure(view_num);
    subplot(4,1,1);
    t = (1:length(ecg))/fs;
    plot(t, ecg);grid on;hold on;
    t_R_index = R_index/fs;
    plot(t_R_index,ecg(R_index),'r*');grid on;
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('ecg信号','FontSize',17);

    subplot(4,1,2);
    plot(t,reg);grid on;hold on;
    t_Imped_index = Imped_index/fs;
    plot(t_Imped_index,reg(Imped_index),'r*');grid on;
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg信号','FontSize',17);

    subplot(4,1,3);
    plot(t,reg);grid on;hold on;
    t_Imped_index = Imped_index2/fs;
    plot(t_Imped_index,reg(Imped_index2),'r*');grid on;
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg信号','FontSize',17);

    subplot(4,1,4);
    plot(t,reg);grid on;hold on;
    t_Imped_index = Imped_index3/fs;
    plot(t_Imped_index,reg(Imped_index3),'r*');grid on;
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg信号','FontSize',17);

end

% Save file
% idx_save_list = [1:N_seg];
% clear dir;
% path_output = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\20211116';
% for idx_save = idx_save_list
%     % 0:Male.  1:Female
%     signal = [[ECG(idx_save,:),SBP(idx_save),DBP(idx_save), fea_hr(idx_save), age];[REG(idx_save,:), H(idx_save), W(idx_save), r_wave_para, 0]];
% 
%     Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
%     index_file = Num_exist_file + 1;
%     filename = num2str(index_file);
%     save(fullfile(path_output, filename),'signal','-v7.3');
% end



