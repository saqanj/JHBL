function [phi] = blockySinosudolPhase(J, n, amp, blockdimension)
%BlockySinosudolPhase Sinosudol phase with a random block.
%
%   [phi] = blockySinosudolPhase(J, n)
%   [phi] = blockySinosudolPhase(J, n, amp)
%   [phi] = blockySinosudolPhase(J, n, amp, blockdimension)
%
%   Inputs
%   ------
%   J     : number of images
%   n     : image size             (returns n-by-n matrices)
%   amp   : base amplitude          [default pi/200   ]  radians
%   blockdimension  : dimension of block with val 1     [default 5 ]
%
%   Outputs
%   -------
%   phi  : all phase with blocks           (n×nxJ double,  in (-pi, +pi])
%
%   Usage inside your generator
%   ---------------------------
%       [phi] = blockySinosudolPhase(J, n);   % call *once* before the loop
%       for jj = 1:J
%           phase(:,jj) = reshape(phi(:, :, j), n, n, 1);
%           ...
%       end

% ---- defaults -----------------------------------------------------------
% if nargin < 2 || isempty(amp),  amp  = 3;   end

if nargin < 3 || isempty(amp),  amp  = pi/200;   end
if nargin < 4 || isempty(blockdimension), blockdimension = 5;  end
if blockdimension >= n/2
    display('too big of a block. defaulting to 5')
    blockdimension = 5;
end

% ---- build the spatial template ----------------------------------------
[X, Y] = meshgrid(1:n, 1:n);
template = sin(2*pi*X/n) .* cos(2*pi*Y/n);   % plaid, range (-1,1)

% ---- scale for initial and incremental phases --------------------------
baseframe = amp  * template;
phi = repmat(baseframe, 1, 1, J);
maxStart = n - blockdimension + 1;                % keep block inside image
block = ones(blockdimension, blockdimension);
for jj = 1: J
    r = randi(maxStart); % row
    c = randi(maxStart); % column
    phi(r:r+blockdimension-1, c:c+blockdimension-1, jj) = block;
end
end