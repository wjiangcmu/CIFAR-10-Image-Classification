%% PCA over raw pixel

X = X_train;

target = labelMat(y_train);
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);
[U,~] = pca(X_norm);
    
%%
% clearvars result times
result = []; times = [];
for k = [5 10 20 50 80 100 200 400 600 800 1000 1200]
    uReduce = U(:, 1:k);
    X_pca = X*uReduce;
    
    tic;
    net = patternnet(k);
    [net,tr] = train(net,X_pca',target','UseParallel','yes','UseGPU','yes');
    
    % evaluation
    testX = X_pca(tr.testInd,:)';
    testT = target(tr.testInd,:)';
    testY = net(testX);
    testIndices = vec2ind(testY);
    [c,~] = confusion(testT,testY);
    
    result = [result, 100*(1-c)];
    times = [times, toc];
    
end
%% Result: top 80 pc performs the best
% [5 10 20 50 80 100 200 400 600 800 1000 1200]

% 27.3333   31.1667   34.1667   34.5000   36.0000   36.0000   33.5000
% 27.3333   23.0000   22.1667   20.5000   19.0000