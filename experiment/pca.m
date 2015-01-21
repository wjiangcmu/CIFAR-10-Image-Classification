function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X

[m, ~] = size(X);
sig = 1/m*(X'*X);
[U, S, ~] = svd(sig);

end
