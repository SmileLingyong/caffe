% Creation       : 10-Mar-2018 10:06
% Last Revision  : 10-Mar-2018 10:06
% Author         : Lingyong Smile {smilelingyong@163.com}
% File Type      : Matlab
% 
% This demo is used to learning caffe.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

%% Init 
close all;
clear;
clc;

%% Init SSD
ssd_path = '/home/lly/work/sina/SSD/caffe-ssd/matlab';
solver_path = './net/lenet_train_test_solver.prototxt';
save_path = './model/';
log_path = './log/';
GPU_ID = 0;

addpath(ssd_path, './utils/');
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(GPU_ID);
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

params = set_params();

%% Loading data
if checkMNIST()
    fprintf('MINST dataset has been downloaded already!\n');
else
    fprintf('Downloading MNIST dataset! Please wait a moment!\n');
    downloadMNIST();
end

%% Load data
[trn_data, trn_labels, tst_data, tst_labels] = LoadMNIST();

%% Create log
if ~exist(log_path, 'dir')
    mkdir(log_path);
end
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file = fullfile(log_path, ['train_', timestamp, '.txt']);
diary(log_file);

%% Init net
caffe_solver = caffe.Solver(solver_path);  % create solver object.
iter = 0;
num_samples = numel(trn_labels); % num_samples = 60000
batch_idx = randperm(num_samples); % return random of num_samples, whilch is a matrix of 1x60000,


%% Start traing 
loss_record = [];
while iter < params.max_iter
    % get batch 
    start_idx = mod(iter * params.batch_sz + 1,  num_samples);
    end_idx = mod((iter + 1) * params.batch_sz,  num_samples);
    if end_idx == 0
        end_idx = num_samples;
    end
    if end_idx < start_idx
        crt_batch_idx = [batch_idx(start_idx : end), batch_idx(1 : end_idx)];
    else
        crt_batch_idx = batch_idx(start_idx : end_idx);
    end
    
    net_input_data = [];
    net_input_label = [];
    for idx = 1:params.batch_sz
        % minus image mean value and transfer the channels are standard
        % image pre-precess methods, 104, 117, 123 these three numbers are
        % the most common mean value which were calculated via ILSVRC
        im = trn_data(:, :, crt_batch_idx(idx));
        label = trn_labels(crt_batch_idx(idx));
        img(:, :, 1) = im - params.mean_value(1); % 104
        img(:, :, 2) = im - params.mean_value(2); % 117
        img(:, :, 3) = im - params.mean_value(3); % 123
        img = img(:, :, [3, 2, 1]); % to adapt the [B, G, R] channels
        net_input_data{idx} = img; %#ok<SAGROW>  cell array 1x64
        net_input_label(idx, :) = label; %#ok<SAGROW>  % numeric array, each loop stored 64x1 matrix
    end
    net_input_data = single(cat(4,net_input_data{:})); % cat to a 4-D tensor, 28x28x3x64 [W H C N]
    net_input_data = permute(net_input_data, [2, 1, 3, 4]); % to adapt [H W], now is [H W C N]
    net_input_label = permute(net_input_label, [4, 3, 2, 1]); 
    
    % set input data    
    caffe_solver.net.blobs('data').set_data(net_input_data);
    caffe_solver.net.blobs('label').set_data(net_input_label);
    
    % forward
    caffe_solver.net.forward_prefilled();
    loss = caffe_solver.net.blobs('loss').get_data();
    
%     % backward 
%     caffe_solver.net.blobs('loss').set_diff(loss);
%     caffe_solver.net.backward_prefilled();
    
    rate_now = params.base_lr * params.gamma^(floor(iter / params.step_size));
    caffe_solver.update(single(rate_now));
    
    iter = iter + 1;
    fprintf('iter %6d/%6d: loss: %.2f\n', iter, params.max_iter, loss); % printf loss information
    
     % visualization loss
    if ~mod(iter, params.show_iter)  % Draw a loss curve while trained every 200 times
        loss_record(end + 1) = loss;
    end
    
    if ~mod(iter, params.show_iter)  % Draw a loss curve while trained every 200 times
        X = 1 : numel(loss_record);
        line(X, loss_record);
        pause(0.1);
    end
    
    
end
    
% save
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
caffe_solver.net.save([save_path, 'final_model.caffemodel']);
diary off; 
