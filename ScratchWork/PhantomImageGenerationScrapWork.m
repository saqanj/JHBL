n = 4096;                                 % the image resolution
E = [ ...
  1.0   .6900  .920   0       0      0 ;  % head; the oval white matter
 -0.8   .6624  .874   0      -.0184  0 ;  % the grey matter inside the white matter
 -0.2   .1100  .310  +.22     0     -18 ; % the back eclipse in the right inside the grey matter
 -0.2   .1600  .410  -.22     0     +18 ; % the back eclipse in the left inside the grey matter
  0.1   .2100  .250   0      +.35    0 ;  % don't know exactly, sth related to the upper white matter inside the grey matter
  0.1   .0460  .046   0      +.10    0 ;  % don't know exactly, sth related to the upper white matter inside the grey matter
  0.1   .0460  .046   0      -.10    0 ;  % 
  0.1   .0460  .023  -.08    -.605   0 ;  % the leftmost small ellipse at the bottom
  0.1   .0230  .023   0      -.606   0 ;  % suspect it the middle small ellipse at the bottom 
  0.1   .0230  .046  +.06    -.605   0 ]; % the rightmost small ellipse at the bottom
Q = phantom(E,n);
imshow(Q,[]); colormap(gray)
