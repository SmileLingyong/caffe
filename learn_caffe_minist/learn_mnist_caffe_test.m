% Creation       : 10-Mar-2018 15:30
% Last Revision  : 10-Mar-2018 15:30
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
save_path = './model/';
modle_path = './net/lenet_test.prototxt';
weights_path = './model/final_model.caffemodel';
log_path = './log/';
GPU_ID = 0;

addpath(ssd_path, './utils/');
caffe.reset_all();
caffe.set_mode_gpu;
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
log_file = fullfile(log_path, ['test_', timestamp, '.txt']);
diary(log_file);

%% Init net
net = caffe.Net(modle_path, weights_path, 'test');
iter = 0;
num_samples = numel(tst_labels); % num_samples = 10000
batch_idx = randperm(num_samples); % return random of num_samples, whilch is a matrix of 1x10000,

%% Start testing
loss_record = [];
while iter < params.max_iter
    % get batch
    start_idx = mod((iter * params.batch_sz) + 1, num_samples);
    end_idx = mod( (iter + 1) * params.batch_sz, num_samples);
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
        img(:, :, 1) = im - params.mean_value(1);
        img(:, :, 2) = im - params.mean_value(2);
        img(:, :, 3) = im - params.mean_value(3);
        img = img(:, :, [3, 2, 1]); % to adapt the [B, G, R] channels
        net_input_data{idx} = img;
        net_input_label(idx, :) = label;
    end
    net_input_data = single(cat(4, net_input_data{:})); % cat to a 4-D tensor, 28x28x3x64 [W H C N]
    net_input_data = permute(net_input_data, [2, 1, 3, 4]); % to adapt [H W], now is [H W C N]
    net_input_label = permute(net_input_label, [4, 3, 2,1]);
    
    net.blobs('data').set_data(net_input_data);
    net.blobs('label').set_data(net_input_label);
    
    % forward
    net.forward_prefilled();
    loss = net.blobs('loss').get_data();
    accuracy = net.blobs('accuracy').get_data();
    
    iter = iter + 1;
    fprintf('iter %6d/%6d: accuracy: %.2f  |  loss: %.2f\n', iter, params.max_iter, accuracy, loss); 

end

diary off;


