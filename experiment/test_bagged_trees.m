clear;clc
load('/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/data.mat');
disp('Data is successfully loaded!')

%% HOG feature
X = [];
for i = 1:length(X_train)
    Xi = [];
      for a = [2]
          for b = [8]
%         for b = [4 6 8 10 16] % 8
            e = extractHOG(X_train(i,:),[b b],[a a]);
            Xi = [Xi, e];
        end
    end
    X = [X; Xi];
end

%% Filter
numOfGroupsToFilter = 2;
numOfDeletions = 20;
cvRate = 0.2;
[ X_filtered, y_filtered, ~, ~, ~ ] = dataFilter( X, y_train, numOfGroupsToFilter, numOfDeletions, cvRate );


%%
X = X_filtered; y = y_filtered;
% y = y_train;
cv = cvpartition(y, 'holdout', 0.2);
Xtrain = X(cv.training,:);
Ytrain = y(cv.training,1);
Xtest = X(cv.test,:);
Ytest = y(cv.test,1);

accs = [];
times = [];
for numTrees = [10 50 100 200 300 400 500]
    tic
    bag = TreeBagger(numTrees, Xtrain, Ytrain);
    [ypred, ~] = bag.predict(Xtest);
    accs = [accs mean(Ytest==str2double(ypred))];
    times = [times toc];
    fprintf('Bag of%3.0f trees are successfully learned! \n', numTrees)
end

%% Submit
bag = TreeBagger(500, X_train,y_train);
disp('Ensemble of trees are successfully learned! \n')
[ypred, ~] = bag.predict(X_test);
writeLabels('label.csv', ypred);

%% result
% No PCA, FS, or filter
% times: 8.0700   37.2960   76.6933  155.1092  248.9542  282.0161  364.6600
% accs:  0.3000    0.4325    0.4738    0.5175    0.5075    0.5025    0.5262

% Filter
% times: 4.6599   17.5437   35.9269   70.3891  105.4762  140.5596  174.9330
% accs:  0.3354    0.4410    0.4648    0.4962    0.5013    0.5276    0.5151



