function [features, R_idx, Imped_max, Imped_min, Imped_d_max] = feature_extract(ecg, reg, fs, N_segment_ecg)
    % R�����
    % ��1�ײ��
    dif = diff(ecg,1);    % �����ǰ��
    % �Ҳ��������ֵ
    N_for = floor(length(ecg)/N_segment_ecg); % ����ѭ���Ķ���
    start = 1;            % ��ʼ�ֶ����
    last = N_segment_ecg; % ��ʼ�ֶ��յ�
    difmax_index = [];    % �洢��������ֵ�±�
    for jj = 1:N_for-1
        dif_segment = dif(start:last);       % ecg�����ÿ�������Ƭ��
        [m, index] = max(dif_segment);       % ecg���������ֵ�±�
        difmax_index = [difmax_index, index+start-1];
        start = start + N_segment_ecg;
        last = last + N_segment_ecg;
    end

    N_dif = 0.1*fs;
    R_idx = []; % R-peak���±�
    for jj = difmax_index
        % ����ecgÿ�����Ƭ�Σ�ȡ���������ֵ�±��ǰ��0.1s
        if(jj-N_dif<=0)
            start = 1;
        else
            start = jj-N_dif;
        end 
        x_segment = ecg(start:jj+N_dif);  
        [m, index] = max(x_segment);
        R_idx = [R_idx, index+jj-N_dif-1];
    end
    if (R_idx(1) <= 0)
        R_idx(1) = [];
        N_for = N_for - 1;
    end

    % ����R��Ѱ�����迹��һ������ֵ
    Imped_max = []; % ���迹���ֵ���±�
    N_segment = 0.5*fs;
    for k = 1:N_for-1
        start = R_idx(k);
        if start<0
            start = 1;
        end
        last = R_idx(k) + N_segment;
        if last>length(reg)
            last = length(reg);
        end
        x_segment = reg(start:last);
        % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
        x_segment_dif = diff(x_segment,1);        % ��֣������ǰ��
        flag = 0;kk = 1;
        while(kk<length(x_segment_dif) && flag == 0)
            if (x_segment_dif(kk)>0 && x_segment_dif(kk+1)<0 && x_segment(kk)>0)
                flag = 1;
                index = kk;
            end
            kk = kk+1;
        end
        Imped_max = [Imped_max, index+start-1]; % ��һ����Сֵ
    end

%     % ����R�������迹���ֵ��ʱ���tao
%     PTT_max_vec = (Imped_max - R_idx)/fs;

    % �������迹��һ������ֵѰ��ǰһ����Сֵ
    Imped_min = [];  % ���迹��Сֵ���±�
    N_segment = 0.5*fs;
    for k = 2:N_for-1
        start = Imped_max(k);
        if start<0
            start = 1;
        end
        last = Imped_max(k) - N_segment;
        if last>length(reg)
            last = length(reg_filted);
        end
        x_segment = reg(last:start);
        % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
        x_segment_dif = diff(x_segment,1);        % ��֣������ǰ��
        flag = 0;kk = length(x_segment_dif)-1;
        while(kk<length(x_segment_dif) && flag == 0)
            if (x_segment_dif(kk)<0 && x_segment_dif(kk+1)>0 && x_segment(kk)<0)
                flag = 1;
                index = kk;
            end
            kk = kk-1;
            if kk < 1
                break;
            end
        end
        Imped_min = [Imped_min, index+last-1]; % ��Сֵ
    end
%     PTT_min_vec = (Imped_max(2:end) - Imped_min)/fs;
    
    % ��1�ײ��
    reg_dif = [0,diff(reg,1)];    % �����ǰ��
    % ����R��Ѱ�����迹��ֵ����ֵ
    Imped_d_max = [];
    N_segment = 0.5*fs;
    for k = 1:N_for-1
        start = R_idx(k);
        if start<0
            start = 1;
        end
        last = R_idx(k) + N_segment;
        if last>length(reg_dif)
            last = length(reg_dif);
        end
        x_segment = reg_dif(start:last);
        % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
        x_segment_dif = diff(x_segment,1);        % ��֣������ǰ��

        flag = 0;kk = 1;
        while(kk<length(x_segment_dif) && flag == 0)
            if (x_segment_dif(kk)>0 && x_segment_dif(kk+1)<0 && x_segment(kk)>0)
                flag = 1;
                index = kk;
            end
            kk = kk+1;
        end
        Imped_d_max = [Imped_d_max, index+start-1]; % ��һ����Сֵ
    end
%     PTT_dif_vec = (Imped_d_max - R_idx )/fs;

%     % ���迹�Ҳ�x%�߶ȵ�
%     Imped_r_x = []; %  ���迹�Ҳ�80%�߶ȵ�
%     height_r_x = 0.8;
%     N_segment = 0.5*fs;
%     for k = 1:N_for-1
%         start = Imped_max(k);
%         if start<0
%             start = 1;
%         end
%         last = Imped_max(k) + N_segment;
%         if last>length(reg)
%             last = length(reg);
%         end
%         x_segment = reg(start:last);
%         % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
%         flag = 0;kk = 1;
%         peak = reg(Imped_max(k));
%         while(kk<length(x_segment) && flag == 0)
%             if (x_segment(kk)>height_r_x*peak && x_segment(kk+1)<height_r_x*peak)
%                 flag = 1;
%                 index = kk;
%             end
%             kk = kk+1;
%         end
%         Imped_r_x = [Imped_r_x, index+start-1]; % ��һ����Сֵ
%     end
%     
%     % ���迹���x%�߶ȵ�
%     Imped_l_x = []; %  ���迹���80%�߶ȵ�
%     height_l_x = 0.8;
%     N_segment = 0.5*fs;
%     for k = 1:N_for-1
%         last = Imped_max(k);
%         if start<0
%             start = 1;
%         end
%         start = Imped_max(k) - N_segment;
%         if start<0
%             start = 1;
%         end
%         x_segment = reg(start:last);
%         % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
%         flag = 0;kk = 1;
%         peak = reg(Imped_max(k));
%         while(kk<length(x_segment) && flag == 0)
%             if (x_segment(kk)<height_l_x*peak && x_segment(kk+1)>height_l_x*peak)
%                 flag = 1;
%                 index = kk;
%             end
%             kk = kk+1;
%         end
%         Imped_l_x = [Imped_l_x, index+start-1]; % ��һ����Сֵ
%     end
    
    % ����Ҫ��ȡ�ĸ߶�
    % �Ҳ�
    height_r_25 = 30;height_r_50 = 50;height_r_75 = 75;height_r_90 = 90;
    [Imped_r_0] = Compute_x_height_right(Imped_max, reg, fs, 0, N_for);
    [Imped_r_25] = Compute_x_height_right(Imped_max, reg, fs, height_r_25, N_for);
    [Imped_r_50] = Compute_x_height_right(Imped_max, reg, fs, height_r_50, N_for);
    [Imped_r_75] = Compute_x_height_right(Imped_max, reg, fs, height_r_75, N_for);
    [Imped_r_90] = Compute_x_height_right(Imped_max, reg, fs, height_r_90, N_for);
    % ���
    height_l_25 = 30;height_l_50 = 50;height_l_75 = 75;height_l_90 = 90;
    [Imped_l_0] = Compute_x_height_left(Imped_max, reg, fs, 0, N_for);
    [Imped_l_25] = Compute_x_height_left(Imped_max, reg, fs, height_l_25, N_for);
    [Imped_l_50] = Compute_x_height_left(Imped_max, reg, fs, height_l_50, N_for);
    [Imped_l_75] = Compute_x_height_left(Imped_max, reg, fs, height_l_75, N_for);
    [Imped_l_90] = Compute_x_height_left(Imped_max, reg, fs, height_l_90, N_for);

    
    
   
%     PTT_max = median(PTT_max_vec);
%     PTT_min = median(PTT_min_vec);
%     PTT_dif_max = median(PTT_dif_vec);


    
    
%     figure(1);
%     subplot(2,1,1);
%     t = (1:length(ecg))/fs;
%     plot(t, ecg);grid on;hold on;
%     t_R_idx = R_idx/fs;
%     plot(t_R_idx,ecg(R_idx),'r*');grid on;
%     xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('ecg�ź�','FontSize',17);
%     subplot(2,1,2);
%     plot(t, reg);grid on;hold on;
%     % ���ֵ
%     t_Imped_max = Imped_max/fs;
%     plot(t_Imped_max,reg(Imped_max),'r*','MarkerSize',10);grid on;
%     % ��Сֵ
%     t_Imped_min = Imped_min/fs;
%     plot(t_Imped_min,reg(Imped_min),'mo','MarkerSize',10);grid on;
%     % ƫ�����ֵ
%     t_Imped_d_max = Imped_d_max/fs;
%     plot(t_Imped_d_max,reg(Imped_d_max),'k^','MarkerSize',10);grid on
%     % �Ҳ����迹�˵�
%     t_Imped_x = Imped_r_0/fs;
%     plot(t_Imped_x,reg(Imped_r_0),'b*','MarkerSize',10);
%     % �Ҳ�25%���迹���ֵ
%     t_Imped_x = Imped_r_25/fs;
%     plot(t_Imped_x,reg(Imped_r_25),'y*','MarkerSize',10);
%     % �Ҳ�50%���迹���ֵ
%     t_Imped_x = Imped_r_50/fs;
%     plot(t_Imped_x,reg(Imped_r_50),'m*','MarkerSize',10);
%     % �Ҳ�75%���迹���ֵ
%     t_Imped_x = Imped_r_75/fs;
%     plot(t_Imped_x,reg(Imped_r_75),'g*','MarkerSize',10);
%     % �Ҳ�90%���迹���ֵ
%     t_Imped_x = Imped_r_90/fs;
%     plot(t_Imped_x,reg(Imped_r_90),'c*','MarkerSize',10);
%     % ������迹�˵�
%     t_Imped_x = Imped_l_0/fs;
%     plot(t_Imped_x,reg(Imped_l_0),'b*','MarkerSize',10);
%     % ���25%���迹���ֵ
%     t_Imped_x = Imped_l_25/fs;
%     plot(t_Imped_x,reg(Imped_l_25),'y*','MarkerSize',10);
%     % ���50%���迹���ֵ
%     t_Imped_x = Imped_l_50/fs;
%     plot(t_Imped_x,reg(Imped_l_50),'m*','MarkerSize',10);
%     % ���75%���迹���ֵ
%     t_Imped_x = Imped_l_75/fs;
%     plot(t_Imped_x,reg(Imped_l_75),'g*','MarkerSize',10);
%     % ���90%���迹���ֵ
%     t_Imped_x = Imped_l_90/fs;
%     plot(t_Imped_x,reg(Imped_l_90),'c*','MarkerSize',10);
%     xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('reg�ź�','FontSize',17);

    
    % ���迹һ�ײ��������ȡ��
    % ����Ҫ��ȡ�ĸ߶�
    % �Ҳ�
    height_r_25 = 30;height_r_50 = 50;height_r_75 = 75;height_r_90 = 90;
    [Imped_d_r_0] = Compute_x_height_right(Imped_d_max, reg_dif, fs, 0, N_for);
    [Imped_d_r_50] = Compute_x_height_right(Imped_d_max, reg_dif, fs, height_r_50, N_for);
    % ���
    height_l_25 = 30;height_l_50 = 50;height_l_75 = 75;height_l_90 = 90;
    [Imped_d_l_0] = Compute_x_height_left(Imped_d_max, reg_dif, fs, 0, N_for);
    [Imped_d_l_50] = Compute_x_height_left(Imped_d_max, reg_dif, fs, height_l_50, N_for);

    
    
%     % ���迹һ�ײ��������ͼ
%     figure(2);
%     subplot(2,1,1);
%     t = (1:length(reg))/fs;
%     plot(t, reg);grid on;hold on;
%     t_idx = Imped_d_max/fs;
%     plot(t_idx,reg(Imped_d_max),'r*');grid on;
%     xlabel('Time (s)','FontSize',17);ylabel('Magnitude','FontSize',17);title('ecg�ź�','FontSize',17);
%     subplot(2,1,2);
%     plot(t, reg_dif);grid on;hold on;
%     % ���ֵ
%     t_Imped = Imped_d_max/fs;
%     plot(t_Imped,reg_dif(Imped_d_max),'r*','MarkerSize',10);grid on;
%     % ��߽�
%     t_Imped_x = Imped_d_l_0/fs;
%     plot(t_Imped_x,reg_dif(Imped_d_l_0),'b*','MarkerSize',10);
%     % ��50%
%     t_Imped_x = Imped_d_l_50/fs;
%     plot(t_Imped_x,reg_dif(Imped_d_l_50),'m*','MarkerSize',10);
%     % �ұ߽�
%     t_Imped_x = Imped_d_r_0/fs;
%     plot(t_Imped_x,reg_dif(Imped_d_r_0),'b*','MarkerSize',10);
%     % ��50%
%     t_Imped_x = Imped_d_r_50/fs;
%     plot(t_Imped_x,reg_dif(Imped_d_r_50),'m*','MarkerSize',10);
%     
    
    
    % �������㣺
    % 1. PTT_MAX,PTT_MIN,PAT
    % PTT_max
    PTT_max_arr = (Imped_max - R_idx)/fs;
    % PTT_min
    PTT_min_arr = (Imped_min - R_idx(2:end))/fs;
    % PAT_max
    PTT_dif_arr = (Imped_d_max - R_idx )/fs;
    % HR
    HR_arr = fs./diff(R_idx,1).*60;    % �����ǰ��
    % t_PTT_max - t_PTT_min
    CT_arr = (Imped_max(2:end) - Imped_min)/fs;
    
    
    % 2. REG�������
    % Width_right_0
    Width_right_0_arr = (Imped_r_0 - Imped_max)/fs;
    % Width_right_25
    Width_right_25_arr = (Imped_r_25 - Imped_max)/fs;
    % Width_right_50
    Width_right_50_arr = (Imped_r_50 - Imped_max)/fs;
    % Width_right_75
    Width_right_75_arr = (Imped_r_75 - Imped_max)/fs;
    % Width_right_90
    Width_right_90_arr = (Imped_r_90 - Imped_max)/fs;
    
    % Width_left_0
    Width_left_0_arr = (Imped_max - Imped_l_0)/fs;
    % Width_left_25
    Width_left_25_arr = (Imped_max - Imped_l_25)/fs;
    % Width_left_50
    Width_left_50_arr = (Imped_max - Imped_l_50)/fs;
    % Width_left_75
    Width_left_75_arr = (Imped_max - Imped_l_75)/fs;
    % Width_left_90
    Width_left_90_arr = (Imped_max - Imped_l_90)/fs;
 
    % Total width
    Width_0_arr = Width_left_0_arr + Width_right_0_arr;
    Width_25_arr = Width_left_25_arr + Width_right_25_arr;
    Width_50_arr = Width_left_50_arr + Width_right_50_arr;
    Width_75_arr = Width_left_75_arr + Width_right_75_arr;
    Width_90_arr = Width_left_90_arr + Width_right_90_arr;

    % Width ratio
    Width_total_arr = (Imped_r_0 - Imped_l_0)/fs;
    Width_25_ratio_arr = Width_25_arr./Width_total_arr;
    Width_50_ratio_arr = Width_50_arr./Width_total_arr;
    Width_75_ratio_arr = Width_75_arr./Width_total_arr;
    Width_90_ratio_arr = Width_90_arr./Width_total_arr;
    
    % Amplitude
    PI_max_arr = reg(Imped_max);
    PI_min_arr = reg(Imped_min);
    PI_d_max_arr = reg(Imped_d_max);
    PP_value_arr = PI_max_arr(2:end) - PI_min_arr;
    
    % PIR (ratio)
    PIR_max_arr = PI_max_arr(2:end)./PI_min_arr;
    PIR_d_max_arr = PI_d_max_arr(2:end)./PI_min_arr;
    
    % Ascending slope
    AS_arr = PP_value_arr./CT_arr;
    % Descending slope
    DS_arr = (reg(Imped_min) - reg(Imped_max(1:end-1)))./((Imped_min - Imped_max(1:end-1))/fs);
    
    
    % ���迹һ�ײ�ֲ�������
    PI_dd_max_arr = reg_dif(Imped_d_max); % һ�ײ�ֵķ�ֵ
    PW_d_0_arr = (Imped_d_r_0 - Imped_d_l_0)/fs;  % һ�ײ�ֵĿ��
    PW_d_50_arr = (Imped_d_r_50 - Imped_d_l_50)/fs;  % һ�ײ��50%�߶ȵĿ��
    PW_ratio_d_arr = PW_d_50_arr./PW_d_0_arr;     % 50%�߶ȵĿ�ȳ����ܿ��
    AS_d_arr = PI_dd_max_arr./((Imped_d_max - Imped_d_l_0)./fs);  % ����б��
    DS_d_arr = PI_dd_max_arr./((Imped_d_max - Imped_d_r_0)./fs);  % �½�б��
    
    
    
    
    % ����8sƬ�ε�ƽ������
    PTT_max = median(PTT_max_arr);
    PTT_min = median(PTT_min_arr);
    PAT_max = median(PTT_dif_arr);
    HR = median(HR_arr);
    CT = median(CT_arr);
    
    Width_right_0 = median(Width_right_0_arr);
    Width_right_25 = median(Width_right_25_arr);
    Width_right_50 = median(Width_right_50_arr);
    Width_right_75 = median(Width_right_75_arr);
    Width_right_90 = median(Width_right_90_arr);

    Width_left_0 = median(Width_left_0_arr);
    Width_left_25 = median(Width_left_25_arr);
    Width_left_50 = median(Width_left_50_arr);
    Width_left_75 = median(Width_left_75_arr);
    Width_left_90 = median(Width_left_90_arr);
    
    Width_total_0 = median(Width_total_arr);
    Width_total_25 = median(Width_25_arr);
    Width_total_50 = median(Width_50_arr);
    Width_total_75 = median(Width_75_arr);
    Width_total_90 = median(Width_90_arr);

    Width_ratio_25 = median(Width_25_ratio_arr);
    Width_ratio_50 = median(Width_50_ratio_arr);
    Width_ratio_75 = median(Width_75_ratio_arr);
    Width_ratio_90 = median(Width_90_ratio_arr);

    PI_max = median(PI_max_arr);
    PI_min = median(PI_min_arr);
    PI_d_max = median(PI_d_max_arr);
    PP_value = median(PP_value_arr);
    
    PIR_max = median(PIR_max_arr);
    PIR_d_max = median(PIR_d_max_arr);
    
    AS = median(AS_arr);
    DS = median(DS_arr);
    
    % ���迹һ�ײ�ֲ�������
    PI_dd_max = median(PI_dd_max_arr);
    PW_d_0 = median(PW_d_0_arr);
    PW_d_50 = median(PW_d_50_arr);
    PW_ratio_d = median(PW_ratio_d_arr);
    AS_d = median(AS_d_arr);
    DS_d = median(DS_d_arr);


    
    
    % ����Ƭ�μ��������
    sd = std(reg);
    kurt = kurtosis(reg);
    skew = skewness(reg);
    En_Ap = ApEn(2, 0.2 * sd, reg);
    En_Sp = SampEn(2, 0.2 * sd, reg);

    
%     PTT_max = mean(PTT_max_vec);
%     PTT_min = mean(PTT_min_vec);
%     PTT_dif_max = mean(PTT_dif_vec);

    features = [PTT_max,PTT_min,PAT_max,CT...
        Width_right_0,Width_right_25,Width_right_50,Width_right_75,Width_right_90,...
        Width_left_0,Width_left_25,Width_left_50,Width_left_75,Width_left_90,...
        Width_total_0,Width_total_25,Width_total_50,Width_total_75,Width_total_90,...
        Width_ratio_25, Width_ratio_50, Width_ratio_75, Width_ratio_90,...
        PI_max, PI_min, PI_d_max, PP_value,...
        PIR_max, PIR_d_max,...
        AS,DS...
        PI_dd_max,PW_d_0,PW_d_50,PW_ratio_d,AS_d,DS_d,...
        sd, skew, kurt, En_Ap, En_Sp
%         HR
        ];

end