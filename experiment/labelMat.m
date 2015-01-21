function [ target ] = labelMat( y_train )
% Format the label vector to 0 and 1 for neural network training
%   y_train is label vector
%   target is label matrix for purpose of nn training

    target = zeros(length(y_train),10);
    for i = 1:length(y_train)
        target(i,y_train(i)) = 1;
    end

end

