function [ nn ] = load_model( file_name )
%LOAD_MODEL Summary of this function goes here
%   Detailed explanation goes here
load(file_name);% w, b, b_prime
% % ========define of network=======
[inputSize, hiddenSize] = size(w);
nn = nnsetup_v2([inputSize, hiddenSize, inputSize]);
nn.Nframe = inputSize/129;
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.W{1}  = [b',w'];
nn.W{2}  = [b_prime',w];

end

