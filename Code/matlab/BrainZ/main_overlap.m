% 本程序仅用于处理脑阻抗数据
% 将.mat格式的脑阻抗数据为8s片段，overlapping6s
% 人眼选择质量好的片段存入数据库

% 修改 path 改变受试者
% 修改 i 改变查看第几次测量的数据
% 修改 idx_save_list 改变要存储的片段的下标
% 存储时，文件名自动按顺序递增

clc; clear all;close all;
fs = 500;
% 按自然顺序读取文件
path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\data_激励频率左右2\1k';
% path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\data_激励频率\1k';

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

% 依次读取每个样本signal
% for i = 1: length(dir_cell)
REG = [];
t_segment = 8;
overlapping = 6;
for i = 3
    % 读取每个样本的ecg和reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    reg = signal(2,:);
    
    T = length(reg)/fs;  % 信号总时间
    move = t_segment - overlapping;
    N_seg = floor((T - t_segment)/(t_segment - overlapping)) + 1;
    % 分段
    idx_start = 1;
    for ii = [1:N_seg]
        % Overlapping
        idx_start = (ii-1) * move * fs + 1;
        idx_end = idx_start + t_segment * fs - 1; 
        
        reg_seg = reg(idx_start:idx_end);
        
        % 提取特征
        REG = [REG;reg_seg];
    end
end

% 看波形
r_wave_para = 250;
for view_num = [1:N_seg]  % 查看一次测量分割为N_seg个片段的波形
    % view_num = 2;
    reg = REG(view_num,:);

    [pks_max,BIOZ_idx_max] = findpeaks(reg,'MinPeakDistance',300);
    [pks_min,BIOZ_idx_min] = findpeaks(-reg,'MinPeakDistance',300);
    
    % 绘图（ecg与ekg在两张图上，查看R波和极小值的检测效果）
    figure(view_num);
    t = (1:length(reg))/fs;
    plot(t,reg);grid on;hold on;
    length_maxBIOZ = length(BIOZ_idx_max);
    length_minBIOZ = length(BIOZ_idx_min);
    len = min(length(BIOZ_idx_max), length(BIOZ_idx_min));
    HI_max = reg(BIOZ_idx_max(1:len));
    HI_min = reg(BIOZ_idx_min(1:len));
    t_BIOZ_idx_max = BIOZ_idx_max(1:len)/fs;
    t_BIOZ_idx_min = BIOZ_idx_min(1:len)/fs;
    plot(t_BIOZ_idx_max,HI_max,'r*');grid on;
    plot(t_BIOZ_idx_min,HI_min,'r*');grid on;
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg信号','FontSize',17);
end

% % Save file
% idx_save_list = [1:N_seg];
% clear dir;
% path_output = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\激励频率左右2\1k';
% for idx_save = idx_save_list
%     signal = REG(idx_save,:);
% 
%     Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
%     index_file = Num_exist_file + 1;
%     filename = num2str(index_file);
%     save(fullfile(path_output, filename),'signal','-v7.3');
% end



