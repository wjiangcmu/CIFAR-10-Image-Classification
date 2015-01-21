clear;clc
load('/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/data.mat');
disp('Data is successfully loaded!')

%% HOG feature extraction: different cell size and block size
X = [];
for i = 1:length(X_train)
    e = extractHOG(X_train(i,:),[b b],[a a]);
    X = [X; e];
end

%% feature selection
infogain = '/Users/Wei/Desktop/Fall2014/10-601 Intro to Machine Learning/project/feature selection/infogain.txt';
fs = readSelectedFeature(infogain);
X = X_train(:,fs(1:200));

%% PCA
k = 100;
X3 = usePCA(X, k);


%% Filter
numOfGroupsToFilter = 3;
numOfDeletions = 20;
cvRate = 0.3;
[ X_filtered, y_filtered, ~, ~, ~ ] = dataFilter( X, y_train, numOfGroupsToFilter, numOfDeletions, cvRate );

%%
% X = X_filtered;
% target = labelMat(y_filtered);

target = labelMat(y_train);

times = [];
result = [];
for k = [5 10 50 100 200 300 500]
    tic;
    net = patternnet(k);
    [net,tr] = train(net,X',target','UseParallel','yes','UseGPU','yes');
    
    % evaluation
    testX = X(tr.testInd,:)';
    testT = target(tr.testInd,:)';
    testY = net(testX);
    testIndices = vec2ind(testY);
    [c,~] = confusion(testT,testY);
    
    result = [result, 100*(1-c)];
    times = [times, toc];
end

%%
% No PCA, FS, or filter
% [5 10 50 100 200 300 500]
% times: 1.5804    1.5455    1.6053    2.3151    3.2528    4.7902   8.5645
% result: 41.1667   45.5000   44.8333   48.6667   48.3333   47.8333   45.6667

% With filter: 3 groups, 20 deletion, 0.3 cv rate
% [5 10 50 100 200 300 500]
% times: 1.6327    1.4047    1.7693    2.3402    3.3347     4.6724    6.7636
% result: 38.8333   48.1667   49.0000   52.0000   45.8333   47.8333   43.0000
