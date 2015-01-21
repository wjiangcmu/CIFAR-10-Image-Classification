function [ result ] = usePCA( X, k )
% Return the top k principle components
%   X: data
%   k: choice of number of principle components
%   result: top k principle components

    mu = mean(X);
    X_norm = bsxfun(@minus, X, mu);
    sigma = std(X_norm);
    X_norm = bsxfun(@rdivide, X_norm, sigma);
    [U,~] = pca(X_norm);
    uReduce = U(:, 1:k);
    result = X*uReduce;
    
end

