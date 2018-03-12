% Creation       : 06-Mar-2018 17:08
% Last Revision  : 06-Mar-2018 17:08
% Author         : Lingyong Smile {smilelingyong@163.com}
% File Type      : Matlab
% 
% This function is used to set params.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

function params = set_params()
    params.batch_sz = 64;
    params.net_input_sz = [28, 28, 1];
    params.max_iter = 5000; 
    params.mean_value = [104, 117, 123]; % mean valuse
    params.base_lr = 0.000001; % base learning rate
    params.gamma = 0.1; % update current base learning rat
    params.step_size = 100000; % update current base learning rate
    params.show_iter = 200;
    params.test_iter = 10;
    
end