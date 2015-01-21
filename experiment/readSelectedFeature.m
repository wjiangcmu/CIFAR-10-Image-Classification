function [ featureIdx ] = readSelectedFeature( input_file )
% Read the text file which obtained from feature selection tool in Weka
%   input_file: the fileID of text which contains selected index of
%   attributes
%   featureIdx: the index of selected attributes as a vector

    fid_train = fopen(input_file);
    A = textscan(fid_train,'%s','Delimiter',','); 
    infor = A{1}(1:end);

    for i = 1:length(infor)
        a = infor(i);
        m = strfind(a,':');
        n = m{1};
        if ~isempty(n)
            infor(i) = [];
        end
    end

    featureIdx = zeros(size(infor));
    for i = 1:length(infor)
        featureIdx(i) = str2num(infor{i});
    end
end

