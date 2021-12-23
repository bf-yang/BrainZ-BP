% 将原始记录的脑阻抗数据进行预处理，不处理ECG
% 将预处理后的脑阻抗存在signal矩阵，并存储在目标文件夹中
% 每次更改新的原始数据路径path_base_input
% 处理后的数据按已有文件个数排序，存储在目标文件夹path_output中

close all; clc; clear;
path_base_input = 'F:\Project-342B\血压预测\Data\激励频率左右2\1k';
path_output = 'F:\Project-342B\血压预测\BloodPressure_Prediction\data_激励频率左右2\1k';
Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
filelist = dir(fullfile(path_base_input, '*.'));
for i = 3:length(filelist)
    % 读取数据
    path = fullfile(path_base_input, filelist(i).name);
    data_arr_3ch = importdata(path);
    signal = signal_compute(data_arr_3ch);
    
    % 存储数据
    index = Num_exist_file + str2num(filelist(i).name);
    filename = num2str(index);
    save(fullfile(path_output, filename),'signal','-v7.3');
end
