% ����vs��vr�������迹
% data_arr_3ch��ԭʼ�������ݵ�����ͨ��
% ��һͨ�����ܵ�ѹ vs
% �ڶ�ͨ�����������˵�ѹ vr
% ����ͨ����ECG

function signal = signal_compute(data_arr_3ch)
    % ���ص�signal��һ����ecg���ڶ�����reg
    fs_raw = 100000; % ����Ƶ��
    
    v_s_raw = data_arr_3ch(:,1); 
    v_r_raw = data_arr_3ch(:,2); 

    N = 200;        % �ֶε�����ÿ��200�㣬2���룩
    fs = fs_raw/N;  % �ֶκ�Ĳ���Ƶ�ʣ�500Hz��

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

        A_vs = [A_vs, As];
        A_vr = [A_vr, Ar];
        Phi_vs = [Phi_vs, Phis];
        Phi_vr = [Phi_vr, Phir];
    end
    tmp1 = A_vs./A_vr;
    tmp2 = Phi_vs - Phi_vr;
    tmp3 = zeros(1, length(tmp2));
    for ii = 1:length(tmp2)
        if(tmp2(ii) > 2*pi)
            tmp2(ii) = tmp2(ii) - 2*pi;
        end
        tmp3(ii) = tmp1(ii)*exp(tmp2(ii).*i);
    end
    Imped = (tmp3 - 1)*10000;  % ����������迹�źţ����迹��

    % ���迹��ģ
    Imped_abs = abs(Imped);

    % 2. ���ĵ��źŷֶ�ƽ�����˳�10kHz���ţ�
    ecg_raw = data_arr_3ch(:,3); 
    % �ֶ�ƽ����
    ecg_seg = [];
    for ii = 1:length(ecg_raw)/N
        point_start = 1 + (ii-1)*N;
        point_end = ii*N;
        ecg_segment = ecg_raw(point_start:point_end); % 100����

        % ���ֵ
        ecg_segment_mean = mean(ecg_segment);
        ecg_seg = [ecg_seg, ecg_segment_mean];
    end

    % 3.ƽ���˲�
    % (1). REG
    Sx_Imped = smoothf(Imped_abs,3,10*100+1);
    Imped_filted = Imped_abs - Sx_Imped;  % ƽ���˲�������迹
    % (2). ECG
    Sx_ecg = smoothf(ecg_seg,3,10*100+1);
    ecg = ecg_seg - Sx_ecg;            % ƽ���˲����ecg

    % 4.���迹��ͨ�˲�
    f1 = 0.5;
    f2 = 10;
    b = fir1(1000,[f1/fs f2/fs]);
    reg = filtfilt(b, 1, Imped_filted);

    % 5.ecg��ͨ�˲�
    f1 = 0.1;
    f2 = 50;
    b = fir1(1000,[f1/fs f2/fs]);
    ecg = filtfilt(b, 1, ecg);

    signal = [ecg; reg];
end