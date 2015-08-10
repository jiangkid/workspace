function [float_score,binary_score] = mfcc_codec_test(netL1, netL2)
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

filePath = 'E:\TIMIT_wav_8k\test';
% filePath = 'F:\ÓïÒô¿â\ZTE_speech\clean';
allfiles = find_wav(filePath);
fileNum = size(allfiles, 1);
num = 100;
fileIdx = randperm(fileNum, num);%Ëæ»úÑ¡Ôñ100¸ö
f_pesq = zeros(num,1);
b_pesq = zeros(num,1);


if isfield(nn,'parallel')&& (nn.parallel == 1)
    nn.binary = 0;
    parfor idx = 1: num
        fileName = allfiles(fileIdx(idx),:);
        disp(fileName);
        [speech_in, fs] = audioread(fileName); % read the wavefile
        speech_out_f = mfcc_codec(speech_in, nn);
        f_pesq(idx) = pesq(speech_in, speech_out_f,fs);
    end
    nn.binary = 1;
    parfor idx = 1: num
        fileName = allfiles(fileIdx(idx),:);
        disp(fileName);
        [speech_in, fs] = audioread(fileName); % read the wavefile
        speech_out_b = mfcc_codec(speech_in, nn);
        b_pesq(idx) = pesq(speech_in, speech_out_b,fs);
    end
else
    for idx = 1: num
        fileName = allfiles(fileIdx(idx),:);
        disp(fileName);
        [speech_in, fs] = audioread(fileName); % read the wavefile
        
        nn.binary = 0;
        speech_out_f = mfcc_codec(speech_in, nn);
        try
            f_pesq(idx) = pesq(speech_in, speech_out_f,fs);
        catch
            f_pesq(idx) = [];
        end
        nn.binary = 1;
        speech_out_b = mfcc_codec(speech_in, nn);
        try
            b_pesq(idx) = pesq(speech_in, speech_out_b,fs);
        catch
            b_pesq(idx) = [];
        end
    end
end
float_score.pesq = mean(f_pesq);
binary_score.pesq = mean(b_pesq);

end
