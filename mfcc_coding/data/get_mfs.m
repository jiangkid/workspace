%获取语音数据
clear;
addpath('f:\matlab\common\voicebox');
addpath('f:\workspace\common\');
data_train = 'E:\TIMIT_wav_8k\train';
dataDir1 = 'E:\TIMIT_wav_8k\train\dr1';
dataDir2 = 'E:\TIMIT_wav_8k\train\dr2';
dataDir3 = 'E:\TIMIT_wav_8k\train\dr3';
dataDir4 = 'E:\TIMIT_wav_8k\train\dr4';
% allfiles = char(find_wav(dataDir1),find_wav(dataDir2),find_wav(dataDir3),find_wav(dataDir4));
% allfiles = find_wav(dataDir2);
allfiles = find_wav(data_train);

w = 'M0y'; J = 70; n = 240; 
% inc = n/2; % 50% overlap
inc = 3*n/4; % 25% overlap
% inc = n; % 0% overlap

data_mfs = [];
% scores = zeros(1,100);
for idx = 1:size(allfiles,1)
    fileName = allfiles(idx,:);
    [y, fs] = audioread(fileName);
    TIMIT_MFS = mfs(y, fs, w, J-1, J, n, inc)';
    data_mfs = [data_mfs;TIMIT_MFS];
end
save('../data/mfs_train.mat','data_mfs');
