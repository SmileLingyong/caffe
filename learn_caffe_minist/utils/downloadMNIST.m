% Creation          : 05-Mar-2018 16:28
% Last Reversion    : 05-Mar-2018 16:28
% Author            : Lingyong Smile {smilelingyong@163.com}
% File Type         : matlab
% 
% This function used to download MNIST dataset from the website, you
% can visit the website to get more details:
% http://yann.lecun.com/exdb/mnist/
% 1. train-images-idx3-ubyte.gz     % training data
% 2. train-labels-idx1-ubyte.gz     % training labels
% 3. t10k-images-idx3-ubyte.gz      % testing data
% 4. t10k-labels-idx1-ubyte.gz      % testing labels
% --------------------------------------------------------------
% Lingyong Smile @ 2018

function [] = downloadMNIST()

save_path = './mnist/';         % save path of mnist dataset
if ~exist(save_path, 'dir')     % if not exist the save directory then make it
    mkdir(save_path);
end

mnist_dataset_website = 'http://yann.lecun.com/exdb/mnist/';        % the mnist dataset webset
file_name = {'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz' ...
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'};     % 4 files of mnist dataset which contains {trn_data, trn_label, tst_data, tst_label}

for idx = 1:length(file_name);
    URL = sprintf([mnist_dataset_website, file_name{idx}]);   % current url of file
    fprintf(1,'Downloading %s...\n',file_name{idx});
    [f, status] = urlwrite(URL, [save_path, file_name{idx}]);   % try to download the files
    gunzip([save_path, file_name{idx}], './mnist/'); % uncompress the GNU zip files
    delete([save_path, file_name{idx}]);   % delete the compressed file, if you want to save the zip files, uncomment this line
    if status == 1;
        fprintf('Finished!\n');
    else
        fprintf('Failed to Download!\n');
    end   
end

end   