% �������迹���ֵ��Ѱ�����迹�Ҳ� x% �ĸ߶ȵ�
% Imped_max�����迹���ֵ�±�
% reg�����迹�ź�
% fs������Ƶ��
% height_r_x��x%�ĸ߶ȣ���λΪ%��������Ϊ30��90
% N_for������ѭ���Ķ���

function [Imped_r_x] = Compute_x_height_right(Imped_max, reg, fs, height_r_x, N_for)
    Imped_r_x = []; %  ���迹�Ҳ�80%�߶ȵ�
    index = 0;
%     height_r_x = 0.8;
    height_r_x = height_r_x / 100; % �ٷ���תΪС��
    N_segment = 0.5*fs;
    for k = 1:N_for-1
        start = Imped_max(k);
        if start<0
            start = 1;
        end
        last = Imped_max(k) + N_segment;
        if last>length(reg)
            last = length(reg);
        end
        x_segment = reg(start:last);
        % ����ź�Ϊ0��ʱ�̼�Ϊ��Сֵʱ��
        flag = 0;kk = 1;
        peak = reg(Imped_max(k));
        while(kk<length(x_segment) && flag == 0)
            if (x_segment(kk)>height_r_x*peak && x_segment(kk+1)<height_r_x*peak)
                flag = 1;
                index = kk;
            end
            kk = kk+1;
        end
        Imped_r_x = [Imped_r_x, index+start-1]; % ��һ����Сֵ
    end
    
end