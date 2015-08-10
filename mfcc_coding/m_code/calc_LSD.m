function [ LSD ] = calc_LSD( mag1, mag2 )
%TEST_LSD Summary of this function goes here
%calculate log spectral distortion (LSD) between mag1 and mag2
LSD = mean(sqrt(mean(10*(log(mag1)-log(mag2))).^2));

end

