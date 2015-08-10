%获取语音数据
clear;
addpath('f:\matlab\common\voicebox');
addpath('f:\workspace\common\');
chinese_train = 'F:\语音库\chinese_speech\train';
allfiles = find_wav(chinese_train);

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
    data_mfs = [data_mfs;TIMIT_MFS(3:end,:)];%去第一二帧
end
save('../data/mfs_chinese_train.mat','data_mfs');
