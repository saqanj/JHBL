function R = create_classical_tv_operator(n)
    e = ones(n,1);
    
    Dx  = spdiags([-e e], [0 1], n,n);      % forward diff, wrap handled later
    Dy  = Dx';                              % for square images
    Gx  = kron(speye(n), Dx);               % vertical direction (rows)
    Gy  = kron(Dy, speye(n));               % horizontal direction (cols)
    R   = [Gx; Gy];
end