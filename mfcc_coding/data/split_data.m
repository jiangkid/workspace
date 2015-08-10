clear;
addpath('F:\workspace\common');
load('mfs_train.mat');%626969
%combine, normalize, split
%%
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs);
save('mfs_train_(N1)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:600000),:);
test_set = mfs_norm(k(600001:610000),:);
valid_set = mfs_norm(k(610001:620000),:);
save('mfs_train_(N1).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 2);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_train_(N2)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:300000),:);
test_set = mfs_norm(k(300001:306000),:);
valid_set = mfs_norm(k(306001:312000),:);
save('mfs_train_(N2).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 4);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_train_(N4)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:130000),:);
test_set = mfs_norm(k(130001:140000),:);
valid_set = mfs_norm(k(140001:150000),:);
save('mfs_train_(N4).mat','train_set','test_set','valid_set');

%%
data_mfs_combine = combineData(data_mfs, 8);
[mfs_norm, mfs_mu, mfs_sigma] = rbm_normalizeData(data_mfs_combine);
save('mfs_train_(N8)_mu_sigma.mat', 'mfs_mu', 'mfs_sigma');
k = randperm(size(mfs_norm,1));
train_set = mfs_norm(k(1:70000),:);
test_set = mfs_norm(k(70001:74000),:);
valid_set = mfs_norm(k(74001:78000),:);
save('mfs_train_(N8).mat','train_set','test_set','valid_set');
