%% wrap up
clear;clc
load('data.mat')
load('X_gist_28.mat')
load('X_gist_44.mat')
load('X_HOG.mat')
load('X_hog_576.mat')
load('X_test_gist.mat')
load('X_test_hog.mat')

% X = double([X_hog X_gist_28]);
% Xtest = double([X_test_hog X_test_gist]);
X = double([X_hog_576 X_gist_44]);
Xtest = double(X_test_hog);
clearvars X_gist_28 X_gist_44 X_hog X_test X_test_gist X_test_hog X_train


%% Info gain
info = '/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/feature selection/info_gist_hog.txt';
fs = readSelectedFeature(info);
X_fs = X(:,fs(1:700));

%% test
Attributes = X_filtered;
Classes = double(y_filtered);
% Attributes = X_filtered;
% Classes = double(y_filtered);
k = 5; % fold
gammas = linspace(0.1,0.5,5);
[ best_acc, best_gamma, time] = useSVM( Attributes, Classes, k, gammas);
%% Submit
submitTest = Xtest;

submitModel = svmtrain(Classes,Attributes,sprintf('-s 0 -t 2 -g %f',best_gamma));
predict_label = zeros(length(submitTest),1);
[predict_label, accuracy, ~] = svmpredict(predict_label, submitTest, submitModel);
writeLabels(sprintf('label_filter_combine_%2.3f.csv',best_gamma), predict_label);


%% Result
% oneR 400:100:800
% 56.4000   57.0250   57.9500   58.5750   58.4500
% 0.5250    0.5250    0.3000    0.3000    0.3000

% info_gain 400:100:800
% 56.2000   57.2750   57.8750   58.4750   58.2500
% 0.5250    0.3000    0.3000    0.3000    0.3000

% small number of features, filter works
% large number of features, fs works at fs = 700