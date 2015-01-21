clear;clc
load('/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/data.mat');
disp('Data is successfully loaded!')
%% HOG feature extraction: different cell size and block size
X_hog_576 = [];
for i = 1:length(X_train)
    e = extractHOG(X_train(i,:),[b b],[a a]);
    X_hog_576 = [X_hog_576; e];
end

%% feature selection
oneR = '/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/feature selection/oneR.txt';
fs = readSelectedFeature(oneR);
X_fs = X(:,fs(1:300));

%% Filter
numOfGroupsToFilter = 3;
numOfDeletions = 30;
partRate = 0.2;
[ X_filtered, y_filtered, removedIdx_target, ~, ~ ] = dataFilter( X, y_train, numOfGroupsToFilter, numOfDeletions, partRate );

%% Classification with filter
best_gammas = [];best_accs = [];
for i = 1:5
    X_filtered = X; y_filtered = y_train;
    Attributes = double(X_filtered);
    Classes = double(y_filtered);
    k = 5;
    gammas = linspace(0.5,3,5);
    [ best_acc, best_gamma] = useSVM( Attributes, Classes, k, gammas);
    best_gammas = [best_gammas, best_gamma];
    best_accs = [best_accs, best_acc];
end
%% Classification with fs
Attributes = X_fs;
Classes = double(y_train);
k = 5;
gammas = linspace(0.1,1.5,5);
[ best_acc, best_gamma] = useSVM( Attributes, Classes, k, gammas);

%% submit
numOfGroupsToFilter = 4;
numOfDeletions = 100;
partRate = 0.2;
[ X_filtered, y_filtered, removedIdx_target, ~, ~ ] = dataFilter(X, y_train, numOfGroupsToFilter, numOfDeletions, partRate );


submitTest = [];
for i = 1:length(X_test)
    Xi = [];
      a = 2; b = 8;
            e = extractHOG(X_test(i,:),[b b],[a a]);
            Xi = [Xi, e];
    submitTest = [submitTest; Xi];
end

Attributes = X_filtered;
Classes = double(y_filtered);
i = 1;
for best_gamma = linspace(0.7,0.9,5)  
    submitModel = svmtrain(double(Classes),Attributes,sprintf('-s 0 -t 2 -g %f',best_gamma));
    predict_label = zeros(length(submitTest),1);
    [predict_label, accuracy, prob_estimates] = svmpredict(predict_label, submitTest, submitModel);
    writeLabels(sprintf('label_%2.2f_filter.csv',best_gamma), predict_label);
    i = i+1
end
%% Notes
% With filtering HOG[2 8]
% 0.7750 57.7889% 52s

% When removing some noise instances from cluster 3 and 4 (both with 10),
% the cv-acc increase by 3%
% best_acc = 57.7889;
% best_gamma = 0.8000;