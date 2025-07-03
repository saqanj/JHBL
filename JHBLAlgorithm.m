rng(1234);

tic
%% JHBL (Joint Hierarchical Bayesian Learning) Algorithm
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023)
% By: Shamsher Tamang & Saqlain Anjum

%% Defining 1-D or 2-D Use Case
is_2D = false;
while true
    response = input('Working in 2-D (1) or 1-D (0)? ');
    if response == 0 || response == 1
        is_2D = logical(response);
        break;
    end
end

%% Defining Important Variables
J = 30; % Num. Images
n = 30; % Value of n for nXn Dimension
max_iterations = 10^3; % For algorithm stopping condition.
max_difference = 10^(-3); % For algorithm stopping condition.

%% Defining Equation Hyperparameters
eta_alpha = 1;
eta_beta = 1;
eta_gamma = 2;

theta_alpha = 10^-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

%% Defining General Linear Transforms, Forward Operator, Data
if is_2D
    R = create_tv_operator(n);
    % R = create_classical_tv_operator(n);
    % R = eye(n^2, n^2)
    K = size(R, 1);
    F = speye(n^2);
    y = zeros(n^2,J);
    x_ground_truth = zeros(n, n, J);
    noise_dimension = n^2;
else
    R = eye(n);
    K = n;
    F = eye(n);
    y = zeros(n,J);
    x_ground_truth = zeros(n, J);
    noise_dimension = n;
end

sparsity_level = 10; % # of non-zero elements
noise_mean = 0;
noise_sd = 0.1;

idx_pool = randi([1, n], sparsity_level, 1); % list of sparsity_level integers in [1, n].
for j = 1:J
    if is_2D
        curr_truth_j = zeros(n, n);
        idx_rows = mod(idx_pool + j - 1, n) + 1; % shifting pattern for rows
        idx_cols = mod(idx_pool + 2*j - 1, n) + 1; % shifting pattern for columns
        lin_indices = sub2ind([n, n], idx_rows, idx_cols); % linear memory indexing (AI)
        curr_truth_j(lin_indices) = 1; 
        x_ground_truth(:, :, j) = curr_truth_j; 
        noise = noise_mean + noise_sd .* randn(noise_dimension, 1);
        y(:, j) = F * curr_truth_j(:) + noise;
    else
        curr_truth_j = zeros(n, 1);
        curr_truth_j(idx_pool) = 1;
        x_ground_truth(:, j) = curr_truth_j; 
        noise = noise_mean + noise_sd .* randn(noise_dimension, 1);
        y(:, j) = F * curr_truth_j + noise;
        idx_pool(end-1:end) = mod(idx_pool(end-1:end), n) + 1;
    end
end

%% Initialization (Algorithm Step 1)
if is_2D
    x = randn(n^2, J);
    gamma = ones(J - 1, n^2);
else
    x = randn(n, J);
    gamma = ones(J - 1, n);
end

alpha = ones(J, 1);
beta = ones(J, K);

%% Algorithmic Iterations
M = zeros(J, 1);
for j = 1:J
    M(j) = length(y(:, j));
end

for l = 1:max_iterations
    x_old = x;

    % -- alpha update
    for j = 1:J
        alpha(j, :) = (eta_alpha + M(j)/2 - 1) / (theta_alpha + 0.5 * norm(F * x(:, j) - y(:, j))^2);
    end

    % -- beta update
    for j = 1:J
        R_x = R * x(:, j); % K x 1
        beta(j, :) = (eta_beta - 0.5) ./ (theta_beta + 0.5 * (R_x(:)).^2);
    end

    % -- gamma update
    for j = 2:J
        gamma(j-1, :) = (eta_gamma - 0.5) ./ (theta_gamma + 0.5 * (x(:, j-1) - x(:, j)).^2);
    end

    % -- x update
    for j = 1:J
        if j == 1
            change_mask_for_G = diag(gamma(j, :));
            change_mask_term_for_b = diag(gamma(j, :)) * x(:, j+1);
        elseif j == J
            change_mask_for_G = diag(gamma(j-1, :));
            change_mask_term_for_b = diag(gamma(j-1, :)) * x(:, j-1);
        else
            change_mask_for_G = diag(gamma(j-1, :)) + diag(gamma(j, :));
            change_mask_term_for_b = diag(gamma(j-1, :)) * x(:, j-1) + diag(gamma(j, :)) * x(:, j+1);
        end

        B_j = diag(beta(j, :));
        G_j = alpha(j, :)*(F')*F + (R')*B_j*R + change_mask_for_G;
        b_j = alpha(j, :)*(F')*y(:, j) + change_mask_term_for_b;

        r = b_j - G_j * x(:, j);
        for i = 1:5
            G_r = G_j * r;
            step_size = (r') * r / (r' * G_r);
            x(:, j) = x(:, j) + step_size * r;
            r = r - step_size * G_r;
        end
    end

    % terminate loop if absolute and relative change condition is met
    difference = x - x_old;
    absolute_change_norms = vecnorm(difference, 2, 1);
    relative_change_norms = absolute_change_norms ./ vecnorm(x, 2, 1);

    average_absolute_change = sum(absolute_change_norms) / J;
    average_relative_change = sum(relative_change_norms) / J;

    if average_absolute_change < max_difference && average_relative_change < max_difference
        break
    end
end

%% Generating Visualizations
if is_2D
    x_reshaped = reshape(x(:, 1), n, n);
    gt_reshaped = x_ground_truth(:, :, 1);
    figure; imshow(x_reshaped, []); title('Reconstructed Image 1');
    figure; imshow(gt_reshaped, []); title('Ground Truth Image 1');
else
    figure;
    plot(x(:, 1)); % plotting the first column vector

    hold on;

    plot(x_ground_truth(:, 1)); % plotting the first column vector
    hold off;
    legend('Prediction','Ground Truth');

    % Side-By-Side Visualizations
    figure;
    plot(x_ground_truth(:, 1));
    legend('Ground Truth')

    hold on;

    figure;
    plot(x(:, 1));
    legend('Prediction');
end

toc