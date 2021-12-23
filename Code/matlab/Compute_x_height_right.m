% 基于脑阻抗最大值，寻找脑阻抗右侧 x% 的高度点
% Imped_max：脑阻抗最大值下标
% reg：脑阻抗信号
% fs：采样频率
% height_r_x：x%的高度，单位为%，即输入为30或90
% N_for：用于循环的段数

function [Imped_r_x] = Compute_x_height_right(Imped_max, reg, fs, height_r_x, N_for)
    Imped_r_x = []; %  脑阻抗右侧80%高度点
    index = 0;
%     height_r_x = 0.8;
    height_r_x = height_r_x / 100; % 百分数转为小数
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
        % 差分信号为0的时刻即为极小值时刻
        flag = 0;kk = 1;
        peak = reg(Imped_max(k));
        while(kk<length(x_segment) && flag == 0)
            if (x_segment(kk)>height_r_x*peak && x_segment(kk+1)<height_r_x*peak)
                flag = 1;
                index = kk;
            end
            kk = kk+1;
        end
        Imped_r_x = [Imped_r_x, index+start-1]; % 第一个极小值
    end
    
end