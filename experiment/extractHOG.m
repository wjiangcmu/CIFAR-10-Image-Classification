function [ features ] = extractHOG( image, CellSize, BlockSize)
% Extrac HOG feature with choice of cell size and block size

%   Defalt: CellSize = [8 8]; BlockSize = [2 2]
%   Cell: [16 16] [10 10] [8 8]   [6 6] [4 4]
%   Block: [2 2] [4 4]

    image1 = image*255;
    recovered = uint8(reshape(image1,[32,32,3]));
    I = rgb2gray(recovered);
    features = double(extractHOGFeatures(I,'CellSize',CellSize,'BlockSize',BlockSize));
end


