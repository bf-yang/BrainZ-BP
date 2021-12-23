% ��ԭʼ��¼�����迹���ݽ���Ԥ����������ECG
% ��Ԥ���������迹����signal���󣬲��洢��Ŀ���ļ�����
% ÿ�θ����µ�ԭʼ����·��path_base_input
% ���������ݰ������ļ��������򣬴洢��Ŀ���ļ���path_output��

close all; clc; clear;
path_base_input = 'F:\Project-342B\ѪѹԤ��\Data\����Ƶ������2\1k';
path_output = 'F:\Project-342B\ѪѹԤ��\BloodPressure_Prediction\data_����Ƶ������2\1k';
Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
filelist = dir(fullfile(path_base_input, '*.'));
for i = 3:length(filelist)
    % ��ȡ����
    path = fullfile(path_base_input, filelist(i).name);
    data_arr_3ch = importdata(path);
    signal = signal_compute(data_arr_3ch);
    
    % �洢����
    index = Num_exist_file + str2num(filelist(i).name);
    filename = num2str(index);
    save(fullfile(path_output, filename),'signal','-v7.3');
end
