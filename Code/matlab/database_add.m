% 将某受试者的数据增添到Dataset_all这个文件夹中（往总数据库里添加数据）
% 将数据增添到总数据的文件夹中

clear;clc;
path = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\20211116';

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));
nsamples = length(dir_cell); 
for i = 1:nsamples
    % 读取每个样本的ecg和reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    
    clear dir;
    path_output = 'F:\Project-342B\血压预测\BloodPressure_Prediction\Dataset_seg\data_all_new';

    Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
    index_file = Num_exist_file + 1;
    filename = num2str(index_file);
    save(fullfile(path_output, filename),'signal','-v7.3');
end