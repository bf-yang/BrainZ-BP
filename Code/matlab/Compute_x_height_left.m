% �������迹���ֵ��Ѱ�����迹��� x% �ĸ߶ȵ�
% Imped_max�����迹���ֵ�±�
% reg�����迹�ź�
% fs������Ƶ��
% height_r_x��x%�ĸ߶ȣ���λΪ%��������Ϊ30��90
% N_for������ѭ���Ķ���

function [Imped_l_x] = Compute_x_height_left(Imped_max, reg, fs, height_l_x, N_for)
    Imped_l_x = []; %  ���迹�Ҳ�80%�߶ȵ�
    index = 0;
%     height_l_x = 0.5;
    height_l_x = height_l_x / 100; % �ٷ���תΪС��
    N_segment = 0.5*fs;
    for k = 1:N_for-1
        last = Imped_max(k);
        start = Imped_max(k) - N_segment;
        if start<=0
            start = 1;
        end
        x_segment = reg(start:last);
        
        flag = 0;kk = length(x_segment);
        peak = reg(Imped_max(k));
        while(kk>1 && flag == 0)
            if (x_segment(kk)>height_l_x*peak && x_segment(kk-1)<height_l_x*peak)
                flag = 1;
                index = kk;
            end
            kk = kk-1;
        end
        Imped_l_x = [Imped_l_x, index+start-1]; % ��һ����Сֵ
    end
end




% function [Imped_l_x] = Compute_x_height_left(Imped_max, reg, fs, height_l_x, N_for)
%     Imped_l_x = []; %  ���迹�Ҳ�80%�߶ȵ�
%     index=0;
% %     height_l_x = 0.5;
%     height_l_x = height_l_x / 100; % �ٷ���תΪС��
%     N_segment = 0.5*fs;
%     for k = 1:N_for-1
%         last = Imped_max(k);
%         start = Imped_max(k) - N_segment;
%         if start<=0
%             start = 1;
%         end
%         x_segment = reg(start:last);
%         
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
% end