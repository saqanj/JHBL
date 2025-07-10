function x = generalized_complex_lasso(F, y, L, lambda, rho, AH, magnitude_only_sparse)

% Inputs:   F - forward operator in matrix form (leave empty if F is FFT)
%           y - data vector
%           L - sparsifying transform
%           rho - augmented Lagrangian parameter (default .1)
%           AH - adjoint of forward operator (leave empty if F is not FFT)
%           magnitude_only_sparse - 1 if sparsity only in magnitude
%                                   0 otherwise
%
% Output:   x - reconstructed complex-valued vector
%
% Written by Dylan Green (2025)

MAX_ITER = 50;
[k, n] = size(L);

x = zeros(n,1);
z = zeros(k,1);
u = zeros(k,1);
LTheta = speye(k,n);

for ii = 1:MAX_ITER
    if isempty(F) % only true for 2D FFT forward op
        x = (speye(n) + rho*(LTheta'*LTheta))\(AH(y) + rho*LTheta'*(z-u));
    else
        x = (F'*F + rho*(LTheta'*LTheta))\(F'*y + rho*LTheta'*(z-u));
    end
    if magnitude_only_sparse
        Theta = x./abs(x).*speye(length(x));
    else
        Theta = speye(length(x));
    end
    LTheta = L*Theta';
    z = wthresh(real(L*Theta'*x+u),"s",lambda/rho);
    u = real(u + L*Theta'*x - z);
end

end