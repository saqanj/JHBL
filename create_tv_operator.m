function R = create_tv_operator(n)
    e = ones(n,1);
    % Second-order finite difference operator: [-1 2 -1] on offsets [-1, 0, 1]
    D = spdiags([-e 2*e -e], [-1 0 1], n, n);

    I = speye(n);
    D_r1 = kron(I, D);   % row 1
    D_r2 = kron(D, I);   % row 2

    R = [D_r1; D_r2];     % Combined anisotropic second-order TV operator
end
