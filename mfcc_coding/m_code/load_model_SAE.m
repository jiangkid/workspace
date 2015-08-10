function [ nn ] = load_model_SAE( file_name, flag )
%LOAD_MODEL Summary of this function goes here
%   Detailed explanation goes here
load(file_name);% w, b, b_prime
% % ========define of network=======
paramSize = size(params,2);
net_size = [size(params{1},1)];
for i=1:paramSize/2
    net_size = [net_size, size(params{2*i},2)];
end
% net_size = [size(params{1},1) size(params{2},2) size(params{4},2) ...
%     size(params{6},2) size(params{8},2), size(params{10},2), size(params{12},2), size(params{14},2), size(params{16},2)];

nn = nnsetup_v3(net_size);
nn.Nframe = net_size(1)/70;
if nargin == 2 && flag == 1
    nn.output = 'linear';
end
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.threshold = 0.5;
for i=1:paramSize/2
    nn.W{i}  = [params{i*2}',params{i*2-1}'];
end
% nn.W{1}  = [params{2}',params{1}'];
% nn.W{2}  = [params{4}',params{3}'];
% nn.W{3}  = [params{6}',params{5}'];
% nn.W{4}  = [params{8}',params{7}'];
% nn.W{5}  = [params{10}',params{9}'];
% nn.W{6}  = [params{12}',params{11}'];
% nn.W{7}  = [params{14}',params{13}'];
% nn.W{8}  = [params{16}',params{15}'];
end

