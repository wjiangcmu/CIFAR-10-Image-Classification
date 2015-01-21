function [ ] = isComplete()
% Play sound

    load gong.mat;
    soundsc(y,2*Fs);
    clearvars Fs y

end

