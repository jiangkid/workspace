function [float_score,binary_score] = speech_lsd_test(netL1, netL2)
% fileName = 'E:\TIMIT_wav_8k\test\dr1\faks0\si1573.wav';%380∂Œ”Ô“Ù ˝æ›
% dataDir = 'E:\TIMIT_wav_8k\test\dr1\faks0';
dataDir = 'E:\TIMIT_wav_8k\test\dr1';
allfiles = find_wav(dataDir);

if nargin == 1
    nn = netL1;
elseif (nargin == 2) && (isempty(netL2) == 0)
    net_size = [netL1.size(1:2), netL2.size(2)];
    nn = nnsetup_v2([net_size flip(net_size(1:end-1))]);
    nn.Nframe = netL1.Nframe;
    nn.activation_function = 'sigm';
    nn.W{1} = netL1.W{1};
    nn.W{4} = netL1.W{2};
    nn.W{2} = netL2.W{1};
    nn.W{3} = netL2.W{2};
end

M = size(allfiles,1);
f_lsd = zeros(M,1);
b_lsd = zeros(M,1);
for idx = 1:M
    fileName = allfiles(idx,:);
    [speech_in, fs] = audioread(fileName);
    nn.LSD = 1;
    nn.binary = 0;    
    f_lsd(idx) = mfcc_codec(speech_in, nn);    
    nn.binary = 1;
    b_lsd(idx) = mfcc_codec(speech_in, nn);    
end
float_score = mean(f_lsd);
binary_score = mean(b_lsd);
end