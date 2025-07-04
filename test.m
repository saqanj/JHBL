n = 64;
img = zeros(n);

img(1:32, 1:32) = 1;       % top-left block
img(33:64, 1:32) = 2;      % bottom-left block
img(1:32, 33:64) = -1;     % top-right block
img(33:64, 33:64) = 0.5;   % bottom-right block

imagesc(img); colormap(gray); axis image; colorbar; 