clear;
addpath('F:\workspace\common');
load('mfs_chinese_train.mat');%1471902
%combine, normalize, split
test_set_n  = 10000; 
%%
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs);
save('mfs_chinese_train_(N1)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
m = fix(size(mfs_norm,1)/test_set_n)*test_set_n;
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:m-test_set_n*2),:);
test_set = mfs_norm(k(m-test_set_n*2+1:m-test_set_n),:);
valid_set = mfs_norm(k(m-test_set_n+1:m),:);
save('mfs_chinese_train_(N1).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 2);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_chinese_train_(N2)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
m = fix(size(mfs_norm,1)/test_set_n)*test_set_n;
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:m-test_set_n*2),:);
test_set = mfs_norm(k(m-test_set_n*2+1:m-test_set_n),:);
valid_set = mfs_norm(k(m-test_set_n+1:m),:);
save('mfs_chinese_train_(N2).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 4);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_chinese_train_(N4)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
m = fix(size(mfs_norm,1)/test_set_n)*test_set_n;
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:m-test_set_n*2),:);
test_set = mfs_norm(k(m-test_set_n*2+1:m-test_set_n),:);
valid_set = mfs_norm(k(m-test_set_n+1:m),:);
save('mfs_chinese_train_(N4).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 8);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_chinese_train_(N8)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
m = fix(size(mfs_norm,1)/test_set_n)*test_set_n;
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:m-test_set_n*2),:);
test_set = mfs_norm(k(m-test_set_n*2+1:m-test_set_n),:);
valid_set = mfs_norm(k(m-test_set_n+1:m),:);
save('mfs_chinese_train_(N8).mat','train_set','test_set','valid_set');
