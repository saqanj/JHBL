% n = 10;                                 % the image resolution
% E = [ ...
%   1.0   .6900  .920   0       0      0 ;  % head; the oval white matter
%  -0.8   .6624  .874   0      -.0184  0 ;  % the grey matter inside the white matter
%  -0.2   .1100  .310  +.22     0     -18 ; % the back eclipse in the right inside the grey matter
%  -0.2   .1600  .410  -.22     0     +18 ; % the back eclipse in the left inside the grey matter
%   0.1   .2100  .250   0      +.35    0 ;  % don't know exactly, sth related to the upper white matter inside the grey matter
%   0.1   .0460  .046   0      +.10    0 ;  % don't know exactly, sth related to the upper white matter inside the grey matter
%   0.1   .0460  .046   0      -.10    0 ;  % 
%   0.1   .0460  .023  -.08    -.605   0 ;  % the leftmost small ellipse at the bottom
%   0.1   .0230  .023   0      -.606   0 ;  % suspect it the middle small ellipse at the bottom 
%   0.1   .0230  .046  +.06    -.605   0 ]; % the rightmost small ellipse at the bottom
% % the columns of E are documented in https://www.mathworks.com/help/images/ref/phantom.html
% Q = phantom(E,n);
% % imshow(Q,[]); colormap(gray)
% 
% mag = abs(Q) + 10^-3;
% phase = angle(Q);
% 
% imshow(mag, []); colormap(gray)


n      = 256;              % image side length
mag0   = phantom(n) + 10^(-3);   % Shepp-Logan + ε
phase0 = smoothPhase(n, 20, 1);

I0 = mag0 .* exp(1i*phase0);      % complex image 0
imshow(mag0, []); colormap(gray)
% imshow(I0)


% ---- frame 1: slightly sharper phantom + faster phase ripples ----
mag1   = imresize(phantom(0.9,n), [n n]) + 1e-3;  % "zoom" 0.9 → edges fade
phase1 = smoothPhase(n, 8,  2);                   % keep seed different
I1 = mag1 .* exp(1i*phase1);

% ---- frame 2: rotate phantom + add linear phase ramp ----
theta  = 8;                            % degrees
mag2   = imrotate(mag0, theta,  'bicubic', 'crop');
[x,y]  = meshgrid(linspace(-1,1,n));
phaseRamp = 2*pi*0.4*x;                % parameter 0.4 = cycles across FOV
phase2 = phase0 + phaseRamp;
I2 = mag2 .* exp(1i*phase2);



figure;
subplot(2,3,1); imshow(mag0,[]);  title('Mag 0');
subplot(2,3,2); imshow(mag1,[]);  title('Mag 1');
subplot(2,3,3); imshow(mag2,[]);  title('Mag 2');

subplot(2,3,4); imagesc(phase0); axis image off; title('Phase 0'); colormap(hsv); colorbar
subplot(2,3,5); imagesc(phase1); axis image off; title('Phase 1'); colormap(hsv); colorbar
subplot(2,3,6); imagesc(phase2); axis image off; title('Phase 2'); colormap(hsv); colorbar

