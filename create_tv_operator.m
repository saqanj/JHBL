function R = create_tv_operator(n)
    e = ones(n,1);

    % Second-Order Finite Difference Operator: [-1 2 -1] with offsets [-1 0 1]
    D = spdiags([-e 2*e -e], [-1 0 1], n, n);
   
    % Boundary Handling
    D(1,:) = 0;  
    D(end,:) = 0; 

    I = speye(n);
    D_r1 = kron(I, D); 
    D_r2 = kron(D, I);

    R = [D_r1; D_r2];  % Full second-order TV operator
end