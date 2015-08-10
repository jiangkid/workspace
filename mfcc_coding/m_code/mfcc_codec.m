function [ ret_val ] = mfcc_codec( speech_in, nn )
%MFCC_CODEC Summary of this function goes here
%   Detailed explanation goes here
mu = nn.mu;
sigma = nn.sigma;
w = 'M0y'; J = 70; n = 240; 
% inc = n/2; % 50% overlap
inc = 3*n/4; % 25% overlap
% inc = n; % 0% overlap

fs = 8000;
win = hamming(n,'periodic');
if isfield(nn,'mfs')&& (nn.mfs == 1)    
    mfcc_data = mfs(speech_in, fs, w, J-1, J, n, inc)';
else
    mfcc_data = melcepst(speech_in, fs, w, J-1, J, n, inc);
end
[data_m,data_dim] = size(mfcc_data);
N = nn.Nframe;
m = fix(data_m/N);
comb_data = zeros(m, data_dim*N);
for i = 1:m
    comb_data(i,:) = reshape(mfcc_data((i-1)*N+1:i*N,:)', 1, data_dim*N);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(nn.output,'sigm')
    [comb_data_norm,~,~] = normalizeData(comb_data,mu,sigma);
elseif strcmp(nn.output, 'linear')
    [comb_data_norm,~,~] = rbm_normalizeData(comb_data,mu,sigma);
end
if nn.binary == 1
nn_out = codec(nn, comb_data_norm);
else
nn_out = nnff(nn, comb_data_norm,comb_data_norm);
end
% histogram(nn_out.a{3}(:,2:end));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(nn.output,'sigm')
    [comb_data_r] = normalizeData_r(nn_out.a{end},mu,sigma);
elseif strcmp(nn.output, 'linear')
    [comb_data_r,~,~] = rbm_normalizeData(nn_out.a{end},mu,sigma,1);    
end
% [comb_data_r] = normalizeData_r(comb_data_norm,mu,sigma);

mfcc_data_r = zeros(m*N, data_dim);
for i = 1:m
    mfcc_data_r((i-1)*N+1:i*N,:) = reshape(comb_data_r(i,:),data_dim,N)';
end
% mfcc_data_r = mfcc_data;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%recover magnitude
[m,~,~]=melbankm(J,n,fs,0,0.5,'y');
if isfield(nn,'mfs')&& (nn.mfs == 1)
mag_r = exp(mfcc_data_r.');
else
mag_r = exp(irdct(mfcc_data_r.'));    
end
m_MP = pinv(full(m)); % Moore-Penrose pseudoinverse
mag_r = m_MP * mag_r;
mag_r = abs(mag_r);

%线性插值
mag = mag_r';
[frameNum, col] = size(mag);
interpNum = 3;%插值个数
mag_interp = zeros(frameNum+(frameNum-1)*interpNum, n/2+1);
mag_interp(1:(interpNum+1):end,:) = mag;
for frameIdx = 1:(frameNum-1)
    mag_interp((frameIdx*(interpNum+1)-(interpNum-1)):frameIdx*(interpNum+1),:) = interp1([1,interpNum+2], mag(frameIdx:frameIdx+1,:), (2:interpNum+1)); %interp1 行线性插值
end
mag_r = mag_interp';

%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(nn,'LSD')&& (nn.LSD == 1)
    mag = exp(irdct(mfcc_data.'));
    m_MP = pinv(full(m)); % Moore-Penrose pseudoinverse
    mag = m_MP * mag;
    mag = abs(mag);
    ret_val = calc_LSD(mag_r, mag);%return the LSD
else
    [outSpeech,~] = LSEE(mag_r,win,16/3);
    ret_val = outSpeech'; %return the speech
end
if (length(speech_in)/length(outSpeech) > 1.2) || (length(speech_in)/length(outSpeech) < 0.8)
    error('length error');
end
end

