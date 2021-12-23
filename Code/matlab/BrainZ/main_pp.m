% 基于脑阻抗的8s数据库
% 计算脑阻抗的 delta Z (PP value)
% 将每个激励频率下的 delta Z 存入excel

clear all;clc;close all;
path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\激励频率左右2\1k';
% path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\激励频率\1k';

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

PP_value_arr = [];
for ii = 1:length(dir_cell)
% for ii = 19 % 2k
% for ii = 18 % 5k   
% for ii = 18 % 10k   
% for ii = 21 % 15k   
% for ii = 22 % 20k   

    signal_struct = load(fullfile(path, dir_cell{ii}));
    reg = signal_struct.signal;

    [pks_max,BIOZ_idx_max] = findpeaks(reg,'MinPeakDistance',300);
    [pks_min,BIOZ_idx_min] = findpeaks(-reg,'MinPeakDistance',300);

    % 计算PP值
    length_maxBIOZ = length(BIOZ_idx_max);
    length_minBIOZ = length(BIOZ_idx_min);
    len = min(length(BIOZ_idx_max), length(BIOZ_idx_min));
    HI_max = reg(BIOZ_idx_max(1:len));
    HI_min = reg(BIOZ_idx_min(1:len));
    PPvalue_arr = HI_max - HI_min;
    PP_value_arr = [PP_value_arr, median(PPvalue_arr)];

%     % 绘图（ecg与ekg在两张图上，查看R波和极小值的检测效果）
%     fs = 500;
%     t = (1:length(reg))/fs;
%     plot(t,reg, 'linewidth',3);hold on;
%     t_BIOZ_idx_max = BIOZ_idx_max(1:len)/fs;
%     t_BIOZ_idx_min = BIOZ_idx_min(1:len)/fs;
% %     plot(t_BIOZ_idx_max,HI_max,'r*', 'linewidth',3);
% %     plot(t_BIOZ_idx_min,HI_min,'r*', 'linewidth',3);
%     xlabel('Time (s)','FontSize',17);ylabel('Amplitude (Ohm)','FontSize',17);
% %     axis([3,5,-70,70]);

end
median(PP_value_arr)

% xlswrite('F:\Project-342B\血压预测\Code\brain_z\左右2\1k.xls', PP_value_arr); 

