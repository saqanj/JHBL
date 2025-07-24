function [phi0, dPhi] = sinosudolPhase(n, amp, step)
%PLAIDPHASESMOOTH  Deterministic plaid phase with a fixed per-frame drift.
%
%   [phi0, dPhi] = plaidPhaseSmooth(n)
%   [phi0, dPhi] = plaidPhaseSmooth(n, amp, step)
%
%   Inputs
%   ------
%   n     : image size             (returns n-by-n matrices)
%   amp   : base amplitude          [default π/10   ]  radians
%   step  : per-frame increment     [default π/200  ]  radians
%
%   Outputs
%   -------
%   phi0  : initial phase           (n×n double,  in (-amp, +amp])
%   dPhi  : add this to phi each frame for smooth temporal change
%
%   Usage inside your generator
%   ---------------------------
%       [phi, dPhi] = plaidPhaseSmooth(n);   % call *once* before the loop
%       for jj = 1:J
%           phi  = phi + dPhi;               % gentle drift
%           phase(:,jj) = phi(:);
%           ...
%       end

% ---- defaults -----------------------------------------------------------
if nargin < 2 || isempty(amp),  amp  = 3;   end
if nargin < 3 || isempty(step), step = pi/200;  end

% ---- build the spatial template ----------------------------------------
[X, Y] = meshgrid(1:n, 1:n);
template = sin(2*pi*X/n) .* cos(2*pi*Y/n);   % plaid, range (-1,1)

% ---- scale for initial and incremental phases --------------------------
phi0 = amp  * template;      % starting frame
dPhi = step * template;      % per-frame addition
end
