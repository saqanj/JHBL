tic
%% Complex-Valued JHBL (Joint Hierarchical Bayesian Learning) Algorithm
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023) and Complex-Valued Signal Recovery Using
% a Generalized Bayesian LASSO (Green, Lindbloom, & Gelb 2025)
% By: Shamsher Tamang & Saqlain Anjum

%% 1-D or 2-D Use Case
is_2D = false;
while true
    response = input('Working in 2-D (1) or 1-D (0)? ');
    if response == 0 || response == 1
        is_2D = logical(response);
        break;
    end
end

%% Parameters
J = 5;
n = 10;
max_iterations = 1e3;
max_difference = 1e-3;

eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 2;
theta_alpha = 1e-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

if is_2D
    R = create_tv_operator(n);
    K = size(R, 1);
    m = n;
    F = @(x) fft2(x)/sqrt(n*m);
    FH = @(x) ifft2(x)*sqrt(n*m);
    y = zeros(n^2, J);
    x_ground_truth = complex(zeros(n, n, J));
    noise_dimension = n^2;
else
    R = eye(n);
    K = n;
    F = @(x) fft(x)/sqrt(n);
    FH = @(x) ifft(x)*sqrt(n);
    y = zeros(n, J);
    x_ground_truth = complex(zeros(n, J));
    noise_dimension = n;
end

sparsity_level = 8;
noise_mean = 0;
noise_sd = 0.1;

for j = 1:J
    if is_2D
        % Piecewise-Constant Magnitude with Smooth Phase (2-D)
        mag = zeros(n, n);
        mag(1:5, 1:5)   = randi([-2, 2]) + 0.1*randn();
        mag(6:10, 1:5)  = randi([-2, 2]) + 0.1*randn();
        mag(1:5, 6:10)  = randi([-2, 2]) + 0.1*randn();
        mag(6:10, 6:10) = randi([-2, 2]) + 0.1*randn();

        [X, Y] = meshgrid(1:n, 1:n);
        phase = (pi/10 * j) * sin(2*pi*X/n) .* cos(2*pi*Y/n);

        curr_truth_j = mag .* exp(1i * phase);
        x_ground_truth(:, :, j) = curr_truth_j;

        noise = noise_mean + noise_sd * randn(noise_dimension, 1);
        y(:, j) = F(curr_truth_j(:)) + noise;
    else
        % Piecewise-Constant Magnitude with Smooth Phase (1-D)
        mag = [ones(3,1); 2*ones(3,1); -1*ones(4,1)];
        mag = mag + 0.1 * randn(n,1);

        phase = linspace(0, pi/2 + 0.05*j, n)';

        x_ground_truth(:, j) = mag .* exp(1i * phase);

        noise = noise_mean + noise_sd * randn(noise_dimension, 1);
        y(:, j) = F(x_ground_truth(:, j)) + noise;
    end
end

%% Initialization
if is_2D
    x = randn(n^2, J) + 1i * randn(n^2, J);
    gamma = ones(J - 1, n^2);
else
    x = randn(n, J) + 1i * randn(n, J);
    gamma = ones(J - 1, n);
end

alpha = ones(J, 1);
beta = ones(J, K);
M = repmat(noise_dimension, J, 1);

%% JHBL Iterations
for l = 1:max_iterations
    x_old = x;

    for j = 1:J
        alpha(j) = (eta_alpha + M(j) - 1) / (theta_alpha + norm(F(x(:, j)) - y(:, j))^2);
    end

    for j = 1:J
        R_x = R * x(:, j);
        beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);
    end

    for j = 2:J
        gamma(j-1, :) = eta_gamma ./ (theta_gamma + abs(x(:, j-1) - x(:, j)).^2);
    end

    for j = 1:J
        if j == 1
            change_mask_G = diag(gamma(j, :));
            change_term_b = gamma(j, :)' .* x(:, j+1);
        elseif j == J
            change_mask_G = diag(gamma(j-1, :));
            change_term_b = gamma(j-1, :)' .* x(:, j-1);
        else
            change_mask_G = diag(gamma(j-1, :) + gamma(j, :));
            change_term_b = gamma(j-1, :)' .* x(:, j-1) + gamma(j, :)' .* x(:, j+1);
        end

        B_j = diag(beta(j, :));
        G_j = alpha(j)*FH(F(1)) + R'*B_j*R + change_mask_G;
        b_j = alpha(j)*FH(y(:, j)) + change_term_b;

        r = b_j - G_j * x(:, j);
        for i = 1:5
            G_r = G_j * r;
            step_size = (r') * r / (r' * G_r);
            x(:, j) = x(:, j) + step_size * r;
            r = r - step_size * G_r;
        end
    end

    difference = x - x_old;
    abs_change = vecnorm(difference, 2, 1);
    rel_change = abs_change ./ vecnorm(x, 2, 1);

    if mean(abs_change) < max_difference && mean(rel_change) < max_difference
        break
    end
end

%% Visualizations
if is_2D
    x_reshaped = reshape(x(:, 1), n, n);
    gt_reshaped = x_ground_truth(:, :, 1);

    figure; imshow(abs(x_reshaped), []); title('Reconstructed Magnitude (Image 1)');
    figure; imshow(angle(x_reshaped), []); title('Reconstructed Phase (Image 1)');
    figure; imshow(abs(gt_reshaped), []); title('Ground Truth Magnitude (Image 1)');
    figure; imshow(angle(gt_reshaped), []); title('Ground Truth Phase (Image 1)');
else
    figure;
    subplot(2,1,1); plot(abs(x(:,1)), 'LineWidth', 1.5); hold on; plot(abs(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
    legend('Reconstructed Magnitude','Ground Truth Magnitude');
    title('1D Magnitude Comparison');

    subplot(2,1,2); plot(angle(x(:,1)), 'LineWidth', 1.5); hold on; plot(angle(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
    legend('Reconstructed Phase','Ground Truth Phase');
    title('1D Phase Comparison');
end
toc