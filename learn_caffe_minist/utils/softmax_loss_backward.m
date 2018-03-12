% Creation       : 06-Mar-2018 19:08
% Last Revision  : 06-Mar-2018 19:08
% Author         : Lingyong Smile {smilelingyong@163.com}
% File Type      : Matlab
% 
% This function is used to calculate softmax loss.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018
% reference : http://blog.csdn.net/l691899397/article/details/52291909
function conf_loss_struct = softmax_loss_backward(conf_loss_struct)
    conf_loss_struct.pred_diff = conf_loss_struct.pred;
    num_class = size(conf_loss_struct.pred, 1);
    for ii = 0 : num_class-1
        cur_class_idx       = find(conf_loss_struct.label == ii);
        conf_loss_struct.pred_diff(ii+1, cur_class_idx) = conf_loss_struct.pred_diff(ii+1, cur_class_idx) - 1;
    end
end