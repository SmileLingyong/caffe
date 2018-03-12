% Creation          : 05-Mar-2018 16:28
% Last Reversion    : 05-Mar-2018 16:28
% Author            : Lingyong Smile {smilelingyong@163.com}
% File Type         : matlab
% 
% This function used to parse the MNIST dataset, the file format
% information is provided in the website: http://yann.lecun.com/exdb/mnist/
%  TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
% 0004     32 bit integer  60000            number of items
% 0008     unsigned byte   ??               label
% 0009     unsigned byte   ??               label
% ........
% xxxx     unsigned byte   ??               label
% 
% The labels values are 0 to 9.
% TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000803(2051) magic number
% 0004     32 bit integer  60000            number of images
% 0008     32 bit integer  28               number of rows
% 0012     32 bit integer  28               number of columns
% 0016     unsigned byte   ??               pixel
% 0017     unsigned byte   ??               pixel
% ........
% xxxx     unsigned byte   ??               pixel
% 
% Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
% TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
% 0004     32 bit integer  10000            number of items
% 0008     unsigned byte   ??               label
% 0009     unsigned byte   ??               label
% ........
% xxxx     unsigned byte   ??               label
% 
% The labels values are 0 to 9.
% TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
% [offset] [type]          [value]          [description]
% 0000     32 bit integer  0x00000803(2051) magic number
% 0004     32 bit integer  10000            number of images
% 0008     32 bit integer  28               number of rows
% 0012     32 bit integer  28               number of columns
% 0016     unsigned byte   ??               pixel
% 0017     unsigned byte   ??               pixel
% ........
% xxxx     unsigned byte   ??               pixel
%
% It is a simple format to store the data
% -------------------------------------------------------------------------
% Lingyong Smile @ 2018

function [] = parsingMNIST()
    dataset_path = './mnist/';
    file_name = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte' ...
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'};     % 4 files of mnist dataset which contains {trn_data, trn_label, tst_data, tst_label}
    save_path = './data/';
    if ~exist(save_path, 'dir')  
        mkdir(save_path);
    end
    for i = 1:numel(file_name)
        crt_file = fullfile(dataset_path, file_name{i});  %  current file{i} path
        fprintf('Parsing %s\n', crt_file);
        fp = fopen(crt_file, 'r');
        magic = fread(fp, 1, 'int32', 0, 'ieee-be');  % read the first byte magic number to judge file type
        % read one byte every time with precision of int32 skip=0 machineformat='ieee-be'
        if magic == 2051   % this is a image file
            % the first 4 byte are some information about this image file
            numImages = fread(fp, 1, 'int32', 0, 'ieee-be'); % the num of images
            numRows = fread(fp, 1, 'int32', 0, 'ieee-be'); % the size of images row
            numCols = fread(fp, 1, 'int32', 0, 'ieee-be'); % the size of images col
            images = fread(fp, inf, 'unsigned char'); % read the rest data with precision='unsigned char' 
            images = reshape(images, numCols, numRows, numImages); % save the data using .mat format
            images = permute(images,[2 1 3]);  % using permute rearrange dimensions of N-D array 
            save([save_path, file_name{i}, '.mat'], 'images');
        elseif magic == 2049 % this is a label file
            numLabels = fread(fp, 1, 'int32', 0, 'ieee-be'); % the num of labels
            labels = fread(fp, inf, 'unsigned char');
            save([save_path, file_name{i}, '.mat'], 'labels'); % save the data using .mat format
        else
            error('Wrong files!');
        end                
        fclose(fp);
    end
    fprintf('Parsing Done!\n');
end