function R = create_tv_operator(n)
    e = ones(n,1);
    % First-Order Finite Difference Operator
    D = spdiags([-e e], [0 1], n, n); D(end,end) = 0;

    I = speye(n);
    D_r1 = kron(I, D);   % Row 1 Entry
    D_r2 = kron(D, I);   % Row 2 Entry

    R = [D_r1; D_r2];     % Total Variation operator
end