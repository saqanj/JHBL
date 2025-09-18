function changeMap = icd_batch(y,win)
%ICD_BATCH  Vectorised interface for y (n²×J) → x (n²×J)
%
%   y   : n²×J matrix, y(:,j) is the j‑th n×n image (vectorised)
%   win : odd window length passed to icd_ml_corr (default 7)
%   x   : n²×J matrix; x(:,1)=1, x(:,j)=ICD map(y1,yj) for j>1
%
[n2,J] = size(y);
n      = round(sqrt(n2));
if n*n ~= n2, error('Input y rows must be a perfect square (n²)'); end
if nargin<2, win = 7; end

changeMap       = ones(n2, J-1, 'double');   % pre‑fill column 1 with ones
for j = 1:J-1
    ref = reshape(y(:,j), n, n);        % reshape reference to prev image
    img = reshape(y(:,j+1), n, n);      % current image
    gamma = icd_ml_corr(ref, img, win); % ICD map
    changeMap(:,j) = gamma(:);          % re‑vectorise
end
end
