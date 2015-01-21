clear;clc
load('/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/data.mat');
disp('Data is successfully loaded!')
%% GIST feature extractioN
X = [];
for i = 1:length(X_train)
    X = [X; extractGIST(X_train(i,:),2,8)];
end


%% Filter
numOfGroupsToFilter = 5;
numOfDeletions = 50;
partRate = 0.2;
[ X_filtered, y_filtered, removedIdx_target, ~, ~ ] = dataFilter( X, y_train, numOfGroupsToFilter, numOfDeletions, partRate );

%% Classification with filter
X_filtered = X; y_filtered = y_train;
Attributes = double(X_filtered);
Classes = double(y_filtered);
partitionRate = 0.3;
gammas = linspace(1,3,5);
[ best_acc, best_gamma, ~ ] = useSVM( Attributes, Classes, partitionRate, gammas);
%% Classification with fs
Attributes = X_fs;
Classes = double(y_train);
partitionRate = 0.2;
gammas = linspace(0.1,1.5,5);
[ best_acc, best_gamma, ~ ] = useSVM( Attributes, Classes, partitionRate, gammas);

%% submit
submitTest = [];
for i = 1:length(X_test)
    submitTest = [submitTest; extractGIST(X_test(i,:),2,8)];
    i
end

X_test_gist = submitTest;


Attributes = X;
Classes = double(y_train);
submitModel = svmtrain(double(Classes),Attributes,sprintf('-s 0 -t 2 -g %f',best_gamma));
predict_label = zeros(length(submitTest),1);
[predict_label, accuracy, prob_estimates] = svmpredict(predict_label, submitTest, submitModel);
writeLabels('label.csv', predict_label);
%% Notes
% With filtering HOG[2 8]
% 0.7750 57.7889% 52s

% When removing some noise instances from cluster 3 and 4 (both with 10),
% the cv-acc increase by 3%
% best_acc = 57.7889;
% best_gamma = 0.8000;