function [ best_acc, best_gamma, time ] = useSVM( Attributes, Classes, k, gammas )
% [ best_acc, best_gamma ] = useSVM( Attributes, Classes, k, gammas )
%
%   This function uses SVM with RBF kernels. "gammas" is a list of gamma 
%   values. A grid search is made with a k-fold cross validation

    
    tic;
    cv = crossvalind('Kfold',length(Classes),k);
    
    acc_gamma = zeros(length(gammas),1);
    j = 1;
    for g = 1:length(gammas)
        accs = zeros(k,1);
        for i = 1:k
            trainData = Attributes(cv~=i,:);
            testData = Attributes(cv==i,:);
            trainLabel = Classes(cv~=i,:);
            testLabel = Classes(cv==i,:);
            model = svmtrain(double(trainLabel),trainData,sprintf('-s 0 -t 2 -g %f',gammas(g))); % RBF
%             model = svmtrain(double(trainLabel),trainData,sprintf('-s 0 -t 0')); % linear
%             model = svmtrain(double(trainLabel),trainData,sprintf('-s 1
%             -t 1 -d %f',gammas(g))); % polynomial
            [~, accuracy, ~] = svmpredict(double(testLabel), testData, model);
            accs(i) = accuracy(1);
            clc;
            fprintf('Finished %2.0f/%2.0f \n',j,k*length(gammas));
            j = j + 1;
        end
        
        acc_gamma(g) = mean(accs);
    end
    
    [best_acc, best_gamma_idx] = max(acc_gamma);
    best_gamma = gammas(best_gamma_idx);
    clc;
    fprintf('Done with evaluation. \n');
    time = toc;
    
end

