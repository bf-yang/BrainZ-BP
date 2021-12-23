% �������迹ʵ�����鲿�;���ֵ
close all;
clc;clear;
fs_raw = 100000; % ����Ƶ��
array_read_1 = importdata('F:\Project-342B\ѪѹԤ��\Data\20211028c\1');

% ������Ư��ר�ã�
% array_read_1 = importdata('F:\Project-342B\ѪѹԤ��\Data\20211116\1');% ECG
% array_read_1 = importdata('F:\Project-342B\ѪѹԤ��\Data\20211102\1');

% 1. ������·��ѹ�źţ������迹
% �������ֶΣ�ÿ�ι��Ʒ��Ⱥ���λ
v_s_raw = array_read_1(:,1); % wwx
v_r_raw = array_read_1(:,2); 

N = 200;       % �ֶε�����ÿ��100�㣬1���룩
fs = fs_raw/N;  % �ֶκ�Ĳ���Ƶ�ʣ�1kHz��


% �ֶ�ƽ����
A_vs = [];
A_vr = [];
Phi_vs = [];
Phi_vr = [];
for ii = 1:length(v_s_raw)/N
    point_start = 1 + (ii-1)*N;
    point_end = ii*N;
    v_s_segment = v_s_raw(point_start:point_end); % 2000����
    v_r_segment = v_r_raw(point_start:point_end);

    % ʹ��100������Ʒ�ֵ����λ
    [As, Phis] = Frequency_estimate(v_s_segment, fs_raw);
    [Ar, Phir] = Frequency_estimate(v_r_segment, fs_raw);
    
%     [As,fs,Phis]=afp_conj_L(v_s_segment',fs_raw,10000-1,10000+1,2,10^(-5));
%     [Ar,fr,Phir]=afp_conj_L(v_r_segment',fs_raw,10000-1,10000+1,2,10^(-5));

    A_vs = [A_vs, As];
    A_vr = [A_vr, Ar];
    Phi_vs = [Phi_vs, Phis];
    Phi_vr = [Phi_vr, Phir];

end
tmp1 = A_vs./A_vr;
tmp2 = Phi_vs-Phi_vr;
tmp3 = zeros(1, length(tmp2));
for ii = 1:length(tmp2)
    if(tmp2(ii) > 2*pi)
        tmp2(ii) = tmp2(ii) - 2*pi;
    end
    tmp3(ii) = tmp1(ii)*exp(tmp2(ii).*i);
end
Imped = (tmp3 - 1)*10000;  % ����������迹�źţ����迹��

% ���迹��ģ
% Imped_abs = abs(Imped);
Imped_abs = real(Imped);
% Imped_abs = imag(Imped);

% 2. ���ĵ��źŷֶ�ƽ�����˳�10kHz���ţ�
ecg_raw = array_read_1(:,3); 
% ecg_raw = array_read_1(:,1); 

% �ֶ�ƽ����
ecg = [];
for ii = 1:length(ecg_raw)/N
    point_start = 1 + (ii-1)*N;
    point_end = ii*N;
    ecg_segment = ecg_raw(point_start:point_end); % 100����

    % ���ֵ
    ecg_segment_mean = mean(ecg_segment);
    
    ecg = [ecg, ecg_segment_mean];
end

% 100����ƽ�������迹��ecg�Ĳ���Ƶ�ʾ�Ϊ1kHz
% ���迹��50Hz���Ž�С������������
% ecg��50Hz�����ܿ�������˹���50HzƵ��
% [A1,f1,p1]=afp_conj_L(ecg,fs_raw/N,50-1,50+1,2,10^(-7));
% n = [1:length(ecg)];
% noise = A1*cos(2*pi*f1*n/1000+p1);
% ecg_de50 = ecg - noise;
ecg_de50 = ecg;

% 4.���迹��ͨ�˲�
f1 = 0.5;
f2 = 10;
b = fir1(1000,[f1/fs f2/fs]);
Imped_filted = filtfilt(b, 1, Imped_abs);

% 5.ecg��ͨ�˲�
f1 = 0.1;
f2 = 50;
b = fir1(1000,[f1/fs f2/fs]);
ecg_filted = filtfilt(b, 1, ecg);


% 3.ƽ���˲�
% (1). REG
Sx_Imped = smoothf(Imped_filted,3,10*100+1);
Imped_filted_fir = Imped_filted - Sx_Imped;  % ƽ���˲�������迹
% (2). ECG
Sx_ecg = smoothf(ecg_filted,3,10*100+1);
ecg_filted = ecg_filted - Sx_ecg;            % ƽ���˲����ecg


figure(926);
t = (1:length(Imped_filted_fir))/fs;
[pks,locs] = findpeaks(Imped_filted_fir,'MinPeakDistance',300);
plot(t, Imped_filted_fir, 'linewidth',3);hold on;grid on;
plot(locs,Imped_filted_fir(locs),'r*');
axis([1,10,-35,35]);
xlabel('Time (s)','FontSize',17);ylabel('Amplitude (mV)','FontSize',17);
legend('Imag part');
set(gca,'FontSize',15);

mean(Imped_filted_fir(locs))
