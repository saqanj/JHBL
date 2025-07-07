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
% % the columns of E are documented in https://www.mathworks.com/help/images/ref/phantom.html

J = 3;
n = 256;              % image resolution
deltaTheta = 20;       % rotation step per image
sigma = 20;           % for phase smoothness
epsMag = 10^(-3);     % for keeping image values > 0 


for jj = 1:J
    % mag{jj} = imrotate(phantom(E, n), theta, 'bicubic', 'crop') + epsMag;
    mag{jj} = abs(phantom(E, n)) + epsMag;
    E(3, 6) = E(3, 6) + deltaTheta;
    E(4, 6) = E(4, 6) - deltaTheta/2;
    phase{jj}      = smoothPhase(n, sigma, jj);

    I{jj}          = mag{jj} .* exp(1i * phase{jj});
end

figure('Name','Synthetic SAR sequence','NumberTitle','off');
for jj = 1:J
    subplot(2,J,jj),     imshow(mag{jj},[]),   title(sprintf('Mag %d',jj));
    subplot(2,J,J+jj),   imagesc(phase{jj}),   axis image off
    title(sprintf('Phase %d',jj)),            colormap(gray); colorbar
end