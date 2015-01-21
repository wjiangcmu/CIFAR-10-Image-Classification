function [ X, y, removedIdx_target, accEachClass, idx_rank ] = dataFilter...
    ( Data, Classes, numOfGroupsToFilter, numOfDeletions, partitionRate )

% This is a supervised method to filter out noise training data. Detailed
% explanation is disscused in the report.
%
% [ X, y, removdedIdx_target, accEachClass, idx_rank ] = dataFilter( Data, Classes, numOfGroupsToFilter, numOfDeletions, partitionRate )
%
%   Data: input data
%
%   Classes: input labels as vector
%
%   numOfGroupsToFilter: choice of number of groups that you want to apply
%   filtering
%
%   numOfDeletions: choice of total number of instances that you want to
%   delete
%
%   partitionRate: rate of partition that split data into training and
%   testing
    
    numOfClasses = length(unique(Classes));
    if numOfGroupsToFilter > numOfClasses || partitionRate >= 1
        return;
    end
    
    % run the ranking process 50 times, find the worst numOfGroups classes
    Cs = zeros(numOfClasses,numOfClasses);
    for j = 1:50 
        P = cvpartition(Classes,'Holdout',partitionRate);
        trainData = Data(P.training,:);
        testData = Data(P.test,:);
        trainLabel = Classes(P.training,:);
        testLabel = Classes(P.test,:);

        % find centroid of each class in trainData
        centroid = zeros(numOfClasses,size(trainData,2));
        for i = 1:numOfClasses
            centroid(i,:) = mean(trainData(find(trainLabel == i),:),1);
        end

        % Using centroid of trainData to cluster testData
        pred_label = zeros(size(testLabel,1),1); %[distance idx]
        for i = 1:size(testData,1)
            dis = zeros(numOfClasses,1);
            for k = 1:numOfClasses
                dis(k) = norm(testData(i,:)-centroid(k,:),3);
            end
            [~, idx] = min(dis);
            pred_label(i) = idx;  
        end

        % count errors for each class in testData using confusionmat  
        [C,~] = confusionmat(double(testLabel),pred_label);
        Cs = Cs+C;
    end
        
        
    accEachClass = zeros(1,10);
    for i = 1:10
        total_i = sum(Cs,2);
        accEachClass(i) = Cs(i,i)/total_i(i);
    end
    [accRate,idx_rank] = sort(accEachClass);
    
    % idx_rank is the rank of accuracy of each class in ascending order
    % the first numOfGroups in idx_rank will be the targe classes to be
    % filtered. In total, there will be numOfDeletion instances to be
    % filtered in these numOfGroups based on the errRate
    
    % find centroid of all data
    centroidAll = zeros(numOfClasses,size(Data,2));
    for i = 1:numOfClasses
        centroidAll(i,:) = mean(Data(find(Classes == i),:),1);
    end
    
    % for each target groups which are to be filtered
    removedIdx_target = [];
    for p = 1:numOfGroupsToFilter
        target = idx_rank(p); accOfTarget = accRate(p);
        numDeletion_target = floor((1-accOfTarget)/sum(1-accRate(1:...
            numOfGroupsToFilter))*numOfDeletions);

        idxOfTarget = find(Classes==target);
        natural_idx_target = zeros(length(idxOfTarget),2); 
        % [distance, index]
        for i = 1:length(idxOfTarget);
            natural_idx_target(i,1) = norm(centroidAll(target,:)-...
                Data(idxOfTarget(i),:),3);
            natural_idx_target(i,2) = idxOfTarget(i);
        end

        [~,ranked_idx_target] = sort(natural_idx_target(:,1),'descend');

        removedIdx_target = [removedIdx_target; natural_idx_target(...
            ranked_idx_target(1:numDeletion_target),2)];
    end
    X = Data; y = Classes;
    X(removedIdx_target,:) = [];
    y(removedIdx_target) = [];
end

