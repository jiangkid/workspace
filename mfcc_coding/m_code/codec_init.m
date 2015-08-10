fs = 8000;
global splitNum J combine minValue maxValue sqNbit
global block1_CB block2_CB block3_CB block4_CB block5_CB block6_CB block7_CB block8_CB
global mfcc_data_max mfcc_data_min w_all counter
counter = 0;
J = 70;
%标量量化
minValue = -58;
maxValue = -8;
sqNbit = 3;

n = 240;

splitNum = 2;% num of split
combine = 1;
load('./mfccCB_54b_v1.mat');
