% Creation          : 05-Mar-2018 16:28
% Last Reversion    : 05-Mar-2018 16:28
% Author            : Lingyong Smile {smilelingyong@163.com}
% File Type         : matlab
% 
% This function used to check if the MNIST dataset has been downloaded.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

function status = checkMNIST()
    status = 1;
    dataset_path = './mnist/';
    file_name = {'train-images-idx3-ubyte', 'train-labels-idx1-ubyte' ...
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'};     % 4 files of mnist dataset which contains {trn_data, trn_label, tst_data, tst_label}
    for i = 1:numel(file_name)
        if ~exist([dataset_path, file_name{i}], 'file')
            status = 0;
        end
    end
end