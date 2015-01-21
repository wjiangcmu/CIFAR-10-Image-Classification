function writeLabels(filename, labels)
% function writeLabels(filename, labels)
%  Writes labels for classification to filename in the format required by
%  Kaggle. (This is from the project starter!)
% 
%  filename - specify the csv's name
%  labels   - m x 1 vector of labels corresponding to each class.
%
%  Example usage:
% 
%  >> writeLabels('foo.csv', [3 5 2,2]');
%  Contents of foo.csv would be:
%    Id,Category
%    1,3
%    2,5
%    3,2
%    4,2
%
    fid = fopen(filename, 'w');
    fprintf(fid, 'Id,Category\n');
    fclose(fid);
    idxs = 1:length(labels(:));
    labels = [idxs', uint32(labels(:))];
    dlmwrite(filename, labels, '-append', 'precision', '%d',...
             'delimiter', ',');
    fprintf('Wrote CSV successfully!\n');
end