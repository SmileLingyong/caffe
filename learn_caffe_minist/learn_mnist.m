% Creation          : 05-Mar-2018 16:28
% Last Reversion    : 05-Mar-2018 16:28
% Author            : Lingyong Smile {smilelingyong@163.com}
% File Type         : matlab
% 
% This demo implemented a very simple Handwritten Digits Recognition
% pipline using MNIST dataset and Random Forest Classifier, it's really a
% common demo for computer vision beginners to realize a very simple
% application of this field.
% -----------------------------------------------------------------------
% Lingyong Smile @ 2018

%% Initialization
close all;
clear;
clc;

%% Downlaod MNIST dataset
addpath('./utils');
if checkMNIST()
    fprintf('MNIST dataset has been downloaded already!\n');
else
    fprintf('Downloading MNIST dataset! Please wait a moment!\n');
    downloadMNIST(); % download and uncompress the MNIST dataset, you must ensure your computer has connected to the Internet. This might takes several minutes.
end

%% Load data
[trn_data, trn_labels, tst_data, tst_labels] =  LoadMNIST();

%% Processing data
% we need to reshape the image from a 28x28 matrix to a 1 * 784 vector for
% training/testing cause the algorithm just support input a vector
trn_data_input = reshape(trn_data, [], numel(trn_labels))';  % set '[]' then the matlab would calculate the dimension num automaticaly
tst_data_input = reshape(tst_data, [], numel(tst_labels))';  % the data would be reshape to a matrix that each row contains one image

%% Training Random Forest Classifier
% Here we are training a random forest model, you can also try a svm or other
% classifiers
fprintf('Start training ... \n');
start_time = tic();
model = TreeBagger(50, trn_data_input, trn_labels);
finished_time = toc(start_time);
fprintf('Training Finished, cost %.2f(s)\n', finished_time);

%% Testing
pred_labels = predict(model, tst_data_input);
pred_labels = str2num(cell2mat(pred_labels)); % convert the pred labels from cell format to numeric
acc = numel(find(pred_labels == tst_labels)) / numel(tst_labels);
fprintf('Accuracy: %.3f\n', acc);

%% Visualization
% select some test images randomly and show the results
figure(1);
set(gcf, 'Position', [500,100,800,800]); % set the window size
num_select = 10;
idx = randperm(numel(tst_labels), num_select * num_select);
pred_idx = pred_labels(idx);
for i = 1:num_select * num_select
    subplot(num_select, num_select, i); % draw a num_select rows num_select colums figure
    imshow(tst_data(:, :, idx(i))); % show the image
    if pred_idx(i) == tst_labels(idx(i)), % prediction is right
        str_title = num2str(pred_idx(i)); % [prediction]
        flag = 'g'; % set the text color be green
    else % prediction is wrong
        str_title = [num2str(pred_idx(i)), '(', num2str(tst_labels(idx(i))), ')']; % [prediction(real label)]
        flag = 'r'; % set the text color be red
    end
    title(str_title, 'Color', flag);
end


