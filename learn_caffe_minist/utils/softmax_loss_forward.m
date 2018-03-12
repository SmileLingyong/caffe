% Creation       : 06-Mar-2018 19:08
% Last Revision  : 06-Mar-2018 19:08
% Author         : Lingyong Smile {smilelingyong@163.com}
% File Type      : Matlab
% 
% This function is used to calculate softmax loss.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

function [ conf_loss_struct ] = softmax_loss_forward( conf_pred_data, conf_gt_data, num_classes )
    % conf_pred_data: 10x64 single， (每一批次（beach）64张图片进行前向传递，然后每张图片得到10个预测结果，共10*64
    % conf_gt_data: 64x1 double     (传入的为net_input_label：即64张图片每张的实际标签)
    % num_classes:1x1, which equal 10
    % reference: 
    %       http://blog.csdn.net/l691899397/article/details/52291909 
    %       http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    %       http://blog.csdn.net/dataningwei/article/details/53925149
    
    %  softmax （分类器，逻辑回归函数, 以下其实就是博客中的第一个的Softmax函数公式）
    max_val             = max( conf_pred_data, [], 1 );  % (1x64single) find max value of each column(total 64 column), 即找到每张图片的最大预测结果
    conf_pred_data      = conf_pred_data - repmat(max_val, num_classes, 1);
    % 其中repmat(max_val, num_classes,1)表示将max_val（1x64）数组，复制10行,得到预测最大值的10×64矩阵
    % 将初始预测值 - 最大预测值, 
    
    exp_conf_pred_data      = exp(conf_pred_data); % 求conf_pred_data的指数值
    sum_exp_conf_pred_data  = sum(exp_conf_pred_data, 1);  % 求每一列的和
    conf_pred_data          = exp_conf_pred_data ./ repmat(sum_exp_conf_pred_data, num_classes, 1);
    % 以上表示使用softmax公式，计算得到64张图片，每一张预测为0-9的概率。
    
    % 然后根据真实标签，10次循环，找其预测的softmax，并通过取负对数，得到其softmax-loss
    conf_loss_struct.pred   = conf_pred_data;
    conf_loss_struct.label  = conf_gt_data;
    
    loss                    = zeros(size(conf_pred_data, 2), 1); % loss is 64*1 single
    for ii = 0 : num_classes-1
        idx                 = find(conf_gt_data == ii);
        loss(idx)           = -log( max(conf_pred_data(ii+1, idx), realmin) );  % 注意理解这里的max(conf_pred_data(ii+1, idx), realmin),其用来 返回指定浮点数类型所能表示的正的最大值，因为这里的conf_pred_data(ii+1, idx)是一个行向量，所以返回的结果依然是一个行向量,
    end
    
    conf_loss_struct.loss   = sum(loss(:));
end  