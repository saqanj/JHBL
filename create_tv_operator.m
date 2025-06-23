function R = create_tv_operator(n)
    e = ones(n,1);
    D = spdiags([-e e], [0 1], n, n); D(end,end) = 0;

    I = speye(n);
    D_h = kron(I, D);   % horizontal
    D_v = kron(D, I);   % vertical

    R = [D_h; D_v];     % Total Variation operator (2n² × n²)
end