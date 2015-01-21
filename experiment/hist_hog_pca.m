%% combine hist, hog and pca
clear;clc
load('data.mat')
load('X_gist_28.mat')
load('X_gist_44.mat')
load('X_HOG.mat')
load('X_test_gist.mat')
load('X_test_hog.mat')

%%
mu = mean(X_train);
X_norm = bsxfun(@minus, X_train, mu);
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);
[U,~] = pca(X_norm);

%%
k = 10;
uReduce = U(:, 1:k);
X_pca = X_train*uReduce;
X = double([X_hog X_gist_28]);

%% svm
Attributes = X;
Classes = double(y_train);
k = 4; % fold
gammas = linspace(0.1,1.5,3);
[ best_acc, best_gamma, time] = useSVM( Attributes, Classes, k, gammas);    
    
%% nn
