clear;
% close all;
addpath(genpath('F:\matlab\DeepLearning\DeepLearnToolbox\trunk'));
addpath('F:\matlab\DeepLearning\speech coding');
load('../data/TIMIT_train_dr1_split.mat')
test_data = test_set(1:100,:);

path_base = 'F:\workspace\speech_coding\src\auto_encoder_out_N1\' ;
% path_base = 'F:\workspace\speech_coding\src\SAE\' ;

% item_str = [path_base,'L1_p0_s1_epoch_1000.mat'];
% netL1 = load_model(item_str);
% netL1_out = nnff(netL1,test_data,test_data);
% x = netL1_out.a{2};
% x = x(:,2:end);%remove bias term

% epoch_list = [1, 5, 10, 50, 100, 500, 1000];%
epoch_list = [1000];
% p_list = [0.1, 0.3, 1, 3, 10];
% p_list = [10];
sigma_list = [0.1, 0.3, 1, 3, 10];
results = cell(length(epoch_list),1);
% for s_idx = 1:length(sigma_list)
% for p_idx = 1:length(p_list)
for idx = 1:length(epoch_list)
%     item_str = [path_base,'SAE_p0_s1_(L1_p0_s1)_(L2_p3_s0.3_(L1_p0_s1))_end.mat'];    
%     item_str = [path_base,'SAE_p0_s1_(L1_p0_s1)_(L2_p10_s0_(L1_p0_s1))_end.mat'];
    item_str = [path_base,'SAE_p0_s1_(L1_p0_s1)_(L2_p0_s1_(L1_p0_s1))_end.mat'];
%     item_str = [path_base,'L2_p0_s1_epoch_',mat2str(epoch_list(idx)),'.mat'];
%     item_str = ['F:\workspace\speech_coding\src\auto_encoder_out\L1_p',mat2str(p_list(p_idx)),'_s0.1_epoch_1.mat'];
%     item_str = ['F:\workspace\speech_coding\src\auto_encoder_out\L1_p0.1_s',mat2str(sigma_list(s_idx)),'_epoch_10.mat'];
%     nn = load_model(item_str);
    nn = load_model_SAE(item_str);
    nn = nnff(nn,test_data,test_data);
%     figure;plot(mean(netL1_out.a{1}(:,2:129)),'*');ylim([0,1]);
%     figure;histogram(netL1_out.a{1});
%     figure(1);plot(mean(netL1_out.a{2}(:,2:501)),'*');ylim([0,1]);    
    figure;histogram(nn.a{3}(:,2:end),50,'Normalization','probability');xlim([0,1]);
%     title(epoch_list(idx));
%     title(p_list(p_idx));
%     title(sigma_list(s_idx));
%     pause(2)
% end
end
% end