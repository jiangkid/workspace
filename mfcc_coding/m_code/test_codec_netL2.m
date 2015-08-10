clear;
addpath('f:\matlab\common\voicebox');
addpath(genpath('F:\matlab\DeepLearning\DeepLearnToolbox\trunk'));
addpath('F:\matlab\DeepLearning\speech coding');
% load('..\data\TIMIT_train_(N11)_mu_sigma.mat');
% load('..\data\TIMIT_train_dr1.dr2_(N5)_mu_sigma.mat');
load('..\data\TIMIT_train_dr1_mu_sigma.mat');
global mu sigma
mu = power_mu;
sigma = power_sigma;
rng('default');rng(0);

% path_base = 'F:\workspace\speech_coding\src\auto_encoder_out_N11\' ;
% path_base = 'F:\workspace\speech_coding\src\auto_encoder_out_N5\' ;
path_base = 'F:\workspace\speech_coding\src\auto_encoder_out_N1\' ;
% % ===========Layer1===============
item_str = [path_base,'L1_p0_s1.mat'];
netL1 = load_model(item_str);

% ===========Layer2===============
% epoch_list = [1, 5, 10, 50, 100, 500, 1000];
% results = cell(length(epoch_list),1);
% for idx = 1:length(epoch_list)
%     item_str = [path_base,'L2_p0_s1_epoch_',mat2str(epoch_list(idx)),'.mat'];
%     if(exist(item_str,'file')==0) continue;end
%     netL2 = load_model(item_str);
%     [v1,v2] = speech_codec_test(netL1, netL2);
%     results{idx} = [v1, v2];
% end
% diary off;

% ===========Layer2===============
p_list = [0, 0.1, 0.3, 1, 3, 10];
sigma_list = [0.1, 0.3, 1, 3, 10];
% p_list = [0.1, 0.3, 1, 3, 10, 30];
% sigma_list = [0];
results = cell(length(p_list),length(sigma_list));
results_pesq = cell(length(p_list), length(sigma_list));
results_segsnr = cell(length(p_list), length(sigma_list));
for p_idx = 1:length(p_list)
for s_idx = 1:length(sigma_list)
    item_str = [path_base,'L2_p',mat2str(p_list(p_idx)),'_s',mat2str(sigma_list(s_idx)),'_(L1_p0_s1).mat'];
    if(exist(item_str,'file')==0) continue;end
    netL2 = load_model(item_str);
%     v1 = 0; v2 = 0;
%     results{p_idx,s_idx} = [v1, v2];
    [float_score, binary_score] = speech_lsd_test(netL1, netL2);
    results{p_idx, s_idx} = [float_score, binary_score];
%     [float_score, binary_score] = speech_codec_test(netL1, netL2);
%     results_pesq{p_idx, s_idx} = [float_score.pesq, binary_score.pesq];
%     results_segsnr{p_idx, s_idx} = [float_score.segsnr, binary_score.segsnr];
end
end
diary off;
