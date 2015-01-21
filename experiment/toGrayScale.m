function [ features ] = toGrayScale( image )
% Convert the given data to a gray scale
%   image: the 1*3072 data given
%   features: a gray scaled vector
    recovered = uint8(reshape(image*255,[32,32,3]));
    I = double(rgb2gray(recovered));
    features = I(:)';

end

