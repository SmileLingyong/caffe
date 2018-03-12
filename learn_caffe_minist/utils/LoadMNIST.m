% Creation          : 05-Mar-2018 16:28
% Last Reversion    : 05-Mar-2018 16:28
% Author            : Lingyong Smile {smilelingyong@163.com}
% File Type         : matlab
% 
% This function used to load MNIST dataset
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

function [trn_data, trn_labels, tst_data, tst_labels] =  LoadMNIST()
    trn_num = 60000; % training samples num
    tst_num = 10000; % testing samples num
    file_name = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte' ...
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'};     % 4 files of mnist dataset which contains {trn_data, trn_label, tst_data, tst_label}
    fprintf('Checking dataset files...\n');
    data_path = './data/';
    status = 1;
    %% Checking .mat data
    for i = 1:numel(file_name)
        fprintf('%s...', file_name{i});
        if ~exist([data_path, file_name{i}, '.mat'], 'file')
            status = 0;
            fprintf(' NOT EXIST!\n');
            break;
        else
            fprintf(' EXIST!\n');
            continue;
        end
    end
    fprintf('Checking Done!\n');
    
    %% Parsing MNIST data
    if ~status
        fprintf('.mat file missing! Start parsing original files!\n');
        parsingMNIST();
    end
    
    %% Loading MNIST data
    for i = 1:numel(file_name)
        data = load([data_path, file_name{i}]);
        if isfield(data, 'images') % data file
            im = data.images;
            im_num = size(im, 3);   % return images number (every image is stored in 3 dimensional matrix )
            if im_num == trn_num, trn_data = im;  % when im_num==60000, thiese are images data
            elseif im_num == tst_num, tst_data = im;
            else error('Loading error, please check the .mat file!');
            end
        elseif isfield(data, 'labels') % labels file
            labels = data.labels;
            labels_num = numel(labels);  % return numbers of labels
            if labels_num == trn_num, trn_labels = labels;
            elseif labels_num == tst_num, tst_labels = labels;
            else error('Loading error, please check the .mat file!');
            end
        end
    end
end