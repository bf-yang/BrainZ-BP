% ����������ڴ������迹����
% ��.mat��ʽ�����迹����Ϊ8sƬ�Σ�overlapping6s
% ����ѡ�������õ�Ƭ�δ������ݿ�

% �޸� path �ı�������
% �޸� i �ı�鿴�ڼ��β���������
% �޸� idx_save_list �ı�Ҫ�洢��Ƭ�ε��±�
% �洢ʱ���ļ����Զ���˳�����

clc; clear all;close all;
fs = 500;
% ����Ȼ˳���ȡ�ļ�
path = 'F:\Project-342B\ѪѹԤ��\BloodPressure_Prediction\data_����Ƶ������2\1k';
% path = 'F:\Project-342B\ѪѹԤ��\BloodPressure_Prediction\data_����Ƶ��\1k';

filelist = dir(fullfile(path, '*.mat'));
dir = struct2cell(filelist); 
dir_cell = sort_nat(dir(1,:));

% ���ζ�ȡÿ������signal
% for i = 1: length(dir_cell)
REG = [];
t_segment = 8;
overlapping = 6;
for i = 3
    % ��ȡÿ��������ecg��reg
    signal_struct = load(fullfile(path, dir_cell{i}));
    signal = signal_struct.signal;
    reg = signal(2,:);
    
    T = length(reg)/fs;  % �ź���ʱ��
    move = t_segment - overlapping;
    N_seg = floor((T - t_segment)/(t_segment - overlapping)) + 1;
    % �ֶ�
    idx_start = 1;
    for ii = [1:N_seg]
        % Overlapping
        idx_start = (ii-1) * move * fs + 1;
        idx_end = idx_start + t_segment * fs - 1; 
        
        reg_seg = reg(idx_start:idx_end);
        
        % ��ȡ����
        REG = [REG;reg_seg];
    end
end

% ������
r_wave_para = 250;
for view_num = [1:N_seg]  % �鿴һ�β����ָ�ΪN_seg��Ƭ�εĲ���
    % view_num = 2;
    reg = REG(view_num,:);

    [pks_max,BIOZ_idx_max] = findpeaks(reg,'MinPeakDistance',300);
    [pks_min,BIOZ_idx_min] = findpeaks(-reg,'MinPeakDistance',300);
    
    % ��ͼ��ecg��ekg������ͼ�ϣ��鿴R���ͼ�Сֵ�ļ��Ч����
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
    xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg�ź�','FontSize',17);
end

% % Save file
% idx_save_list = [1:N_seg];
% clear dir;
% path_output = 'F:\Project-342B\ѪѹԤ��\BloodPressure_Prediction\Dataset_seg\����Ƶ������2\1k';
% for idx_save = idx_save_list
%     signal = REG(idx_save,:);
% 
%     Num_exist_file = length(dir(fullfile(path_output, '*.mat')));
%     index_file = Num_exist_file + 1;
%     filename = num2str(index_file);
%     save(fullfile(path_output, filename),'signal','-v7.3');
% end



