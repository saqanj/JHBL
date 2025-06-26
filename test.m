n = 100;
f = zeros(1, n);

f(1:30) = 2;
f(31:60) = -1;
f(61:100) = 0.5;

plot(f, 'LineWidth', 2);
title('Discrete Piecewise Constant Signal'); grid on;