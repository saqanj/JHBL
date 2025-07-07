function phi = smoothPhase(n, sigma, seed)
% smoothPhase  Smooth 2-D phase field in (-pi,pi]
%   n     : image size
%   sigma : Gaussian blur std-dev (higher ⇒ smoother)
%   seed  : rng seed for reproducibility
%
%   phi   : n-by-n phase matrix (radians)
%
if nargin>2, rng(seed); end
phi = randn(n);                     % white noise
h   = fspecial('gaussian', [n n], sigma);
phi = imfilter(phi, h, 'replicate');% Gaussian blur
phi = pi * phi / max(abs(phi(:)));  % scale to (-pi,pi]
end
