function a = icd_ml_corr(img1,img2,win)
% Maximum–likelihood estimate of the normalised complex correlation
%   img1,img2 : n×n complex (double|single)
%   win       : odd scalar window length (default 7)
%   a         : n×n map of α̂  (0↔big change, 1↔no change)
% Implements Eq.(5.102) of [Curlander & McDonough] over an N×N window.
%
if nargin<3||isempty(win), win=7; end
if mod(win,2)==0, error('win must be odd'); end
img1 = double(img1); img2 = double(img2);
k    = ones(win,win);
% num  = abs( conv2(img1.*conj(img2),k,'same') );                       % |Σ fg*|
num = abs(conv2(img1.*conj(img2),k,'same'));
den  = sqrt( conv2(abs(img1).^2,k,'same') .* ...
             conv2(abs(img2).^2,k,'same') );                          % √(Σ|f|² Σ|g|²)
a    = num ./ (den+eps);                                              % α̂∈[0,1]
end
