function L = sparse_operator(N1,N2,PAORDER)

% Creates a sparsifying transform operator L with zero boundary conditions
%
% Inputs:
%   - N1 = dimension 1
%   - N2 = dimension 2
%   - PAORDER = 0 if signal sparse, 1 if edges sparse
%
% Output:
%   - L = sparsifying transform
%
% Written by Dylan Green (2025)

if N2 == 1
    switch PAORDER
        case {0}
            L = sparse(eye(N1));
        case {1}
            L = sparse(eye(N1) - circshift(eye(N1),-1,2));
            L(1,:) = [1 zeros(1,N1-1)];
    end
else
    switch PAORDER
        case {0}
            L = speye(N1 * N2);
        case {1}
            T = eye(N1);
            T1 = -circshift(eye(N1),-1,2);
            T1(1,end) = 0;
            T2 = -eye(N1);
            L1 = sparse(N1*N2,N1*N2);
            L2 = sparse(N1*N2,N1*N2);
            L1(1:N1,1:N1) = T+T1;
            L2(1:N1,1:N1) = T;
            for ii = 1:N2-1
                L1(ii*N1+1:(ii+1)*N1,ii*N1+1:(ii+1)*N1) = T+T1;
                L2(ii*N1+1:(ii+1)*N1,ii*N1+1:(ii+1)*N1) = T;
                L2(ii*N1+1:(ii+1)*N1,(ii-1)*N1+1:ii*N1) = T2;
            end
            L1 = sparse(L1);
            L2 = sparse(L2);
            L = [L1;L2];
    end     
end