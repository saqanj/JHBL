clear all;
rng(4)

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
J = 3;
n = 100;
max_iterations = 1e4;
max_difference = 1e-3;

eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 2;
theta_alpha = 1e-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

if is_2D
    R = sparse_operator(n,n,1);
    K = size(R, 1);
    m = n;
    F = @(x) reshape(fft2(reshape(x,n,m))/sqrt(n*m),n*m,1);
    FH = @(x) reshape(ifft2(reshape(x,n,m))*sqrt(n*m),n*m,1);
    y = zeros(n^2, J);
    x_ground_truth = complex(zeros(n, m, J));
    noise_dimension = n*m;
else
    e = ones(n, 1);
    R = spdiags([e -e], [0, 1], n-1, n);
    K = size(R, 1);
    F = @(x) fft(x)/sqrt(n);
    FH = @(x) ifft(x)*sqrt(n);
    y = zeros(n, J);
    x_ground_truth = complex(zeros(n, J));
    noise_dimension = n;
end

sparsity_level = 8;
noise_mean = 0;
noise_sd = 0.1;

mag(10:50, 10:50)   = randi([0, 2]);
mag(50:100, 10:50)  = randi([0, 2]);
mag(10:50, 60:100)  = randi([0, 2]);
mag(16:25, 6:25) = randi([0, 2]);

mag = mag + 0.5;

for j = 1:J
    if is_2D
        [X, Y] = meshgrid(1:n, 1:n);
        phase = (pi/10 * j) * sin(2*pi*X/n) .* cos(2*pi*Y/n);
        curr_truth_j = mag .* exp(1i * phase);
        x_ground_truth(:, :, j) = curr_truth_j;
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
        y(:, j) = F(curr_truth_j(:)) + noise;
    else
        mag = [ones(30,1); 2*ones(30+j,1); 1.5*ones(40-j,1)];
        phase = linspace(0, pi/2 + 0.05*j, n)';
        x_ground_truth(:, j) = mag .* exp(1i * phase);
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension, 1));
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
        alpha(j) = (eta_alpha + M(j) - 1) / (theta_alpha + norm(abs(F(x(:, j)) - y(:, j)),2)^2);
    end

    % for j = 1:J
    %     Theta_j = x(:, j) ./ (abs(x(:, j))); % phase calculation
    %     % apply TV only to magnitude
    %     LTheta_j = R .* Theta_j';          
    % 
    %     R_x = LTheta_j * x(:, j);
    %     beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);
    % end

    for j = 2:J
        gamma(j-1, :) = eta_gamma ./ (theta_gamma + abs(x(:, j-1) - x(:, j)).^2);
    end

    for j = 1:J
        Theta_j = x(:, j) ./ (abs(x(:, j))); % phase calculation
        % apply TV only to magnitude
        LTheta_j = R .* Theta_j';
        R_x = LTheta_j * x(:, j);
        beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);

        if j == 1
            change_mask_G = gamma(j, :).';
            change_term_b = gamma(j, :).' .* x(:, j+1);
        elseif j == J
            change_mask_G = gamma(j-1, :).';
            change_term_b = gamma(j-1, :)' .* x(:, j-1);
        else
            change_mask_G = gamma(j-1, :).' + gamma(j, :).';
            change_term_b = gamma(j-1, :).' .* x(:, j-1) + gamma(j, :).'.* x(:, j+1);
        end

        G_j = @(x_var) alpha(j)*FH(F(x_var)) + LTheta_j' * (beta(j,:).' .* (LTheta_j * x_var)) + change_mask_G .* x_var;
        b_j = alpha(j)*FH(y(:, j)) + change_term_b;

        r = b_j - G_j(x(:, j));
        for i = 1:5
            G_r = G_j(r);
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