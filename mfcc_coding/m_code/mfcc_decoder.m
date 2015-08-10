function [sig_r,mfcc_data] = mfcc_decoder(mfcc_data_idx,flag)
%
if nargin<2 flag=0; end
codec_init;
global J  combine mfcc_data_max mfcc_data_min sqNbit
win = hamming(n,'periodic');
[frameNum, col] = size(mfcc_data_idx);
if 1
mfcc_data = zeros(frameNum, J);
%反量化
for idx = 1:frameNum
    mfcc_data(idx,:)  = mfcc_vq_d( mfcc_data_idx(idx,:) );
%     mfcc_data(combine*(idx-1)+1:combine*idx,:) = mfcc_sq_d( mfcc_data_idx(idx,:) );
end
end
%recover magnitude
[m,a,b]=melbankm(J,n,fs,0,0.5,'y');
mag_r = exp(irdct(mfcc_data.'));
m_MP = pinv(full(m)); % Moore-Penrose pseudoinverse
mag_r = m_MP * mag_r;

%线性插值
mag = mag_r';
interpNum = 3;%插值个数
mag_interp = zeros(frameNum+(frameNum-1)*interpNum, n/2+1);
mag_interp(1:(interpNum+1):end,:) = mag;
for frameIdx = 1:(frameNum-1)
    mag_interp((frameIdx*(interpNum+1)-(interpNum-1)):frameIdx*(interpNum+1),:) = interp1([1,interpNum+2], mag(frameIdx:frameIdx+1,:), (2:interpNum+1)); %interp1 行线性插值
end
mag_interp = mag_interp';
% 插值后，相当于
overlap = 16/3;% 

%%recover signal
if flag == 0
    [sig_r,] = LSEE(mag_interp,win,overlap);
else
    [sig_r,] = LSEE_min_pha(mag_interp,win,overlap);
end

end
