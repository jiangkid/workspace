clear;
load('TIMIT_train.mat');
m = size(mfcc_norm,1);
k = randperm(m);
train_set = mfcc_norm(k(1:300000),:);
test_set = mfcc_norm(k(300001:306000),:);
valid_set = mfcc_norm(k(306001:end),:);
save('TIMIT_train_split.mat','train_set','test_set','valid_set');
