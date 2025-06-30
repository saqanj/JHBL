tic
%% JHBL (Joint Hierarchical Bayesian Learning) Algorithm
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023) and Complex-Valued Signal Recovery Using
% a Generalized Bayesian LASSO (Green, Lindbloom, & Gelb 2025)
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
J = 5; % Num. Images
n = 10; % Value of n for nXn Dimension
max_iterations = 10^3; % For algorithm stopping condition.
max_difference = 10^(-3); % For algorithm stopping condition.

%% Defining Equation Hyperparameters
eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 2;

theta_alpha = 10^-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

%% Defining General Linear Transforms, Forward Operator, Data
if is_2D
    R = create_tv_operator(n);
    K = size(R, 1);
    m = n;      % for square images
    F = @(x) fft2(x)/sqrt(n*m);
    FH = @(x) ifft2(x)*sqrt(n*m);
    y = zeros(n^2,J);
    x_ground_truth = zeros(n, n, J);
    noise_dimension = n^2;
else
    R = eye(n);
    K = n;
    F = @(x) fft(x)/sqrt(n);
    FH = @(x) ifft(x)*sqrt(n);
    y = zeros(n, J);
    x_ground_truth = zeros(n, J);
    noise_dimension = n;
end

sparsity_level = 8; % # of non-zero elements
noise_mean = 0;
noise_sd = 0.1;

 for j = 1:J
    if is_2D
        % Piecewise Constant Image Generation 2-D Case
        x_ground_truth(1:5, 1:5, j)   = randi([-2, 2]);  % top-left block
        x_ground_truth(6:10, 1:5, j)  = randi([-2, 2]);  % bottom-left block
        x_ground_truth(1:5, 6:10, j)  = randi([-2, 2]);  % top-right block
        x_ground_truth(6:10, 6:10, j) = randi([-2, 2]);  % bottom-right block
        curr_truth_j = x_ground_truth(:, :, j);

        noise = noise_mean + noise_sd .* randn(noise_dimension, 1);
        y(:, j) = F(curr_truth_j(:)) + noise;
    else
        % Piecewise Constant Image Generation 1-D Case
        x_ground_truth(:,1) = [ones(3,1); 2*ones(3,1); -1*ones(4,1)];        % 3 + 3 + 4 = 10
        x_ground_truth(:,2) = [zeros(2,1); 3*ones(3,1); -2*ones(5,1)];       % 2 + 3 + 5 = 10
        x_ground_truth(:,3) = [1*ones(5,1); -1*ones(5,1)];                   % 5 + 5 = 10
        x_ground_truth(:,4) = [1.5*ones(2,1); 0*ones(3,1); 2*ones(5,1)];     % 2 + 3 + 5 = 10
        x_ground_truth(:,5) = [-1*ones(4,1); 0.5*ones(3,1); -0.5*ones(3,1)]; % 4 + 3 + 3 = 10

        noise = noise_mean + noise_sd .* randn(noise_dimension, 1);
        y(:, j) = F(x_ground_truth(:, j)) + noise;
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
        alpha(j, :) = (eta_alpha + M(j) - 1) / (theta_alpha + norm(F(x(:, j)) - y(:, j))^2);
    end

    % -- beta update
    for j = 1:J
        R_x = R * x(:, j); % K x 1
        beta(j, :) = (eta_beta) ./ (theta_beta + (R_x(:)).^2);
    end

    % -- gamma update
    for j = 2:J
        gamma(j-1, :) = (eta_gamma) ./ (theta_gamma + (x(:, j-1) - x(:, j)).^2);
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
        G_j = alpha(j, :)*(FH(F(1))) + (R')*B_j*R + change_mask_for_G;
        b_j = alpha(j, :)*(FH(y(:, j))) + change_mask_term_for_b;

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
    plot(x(:, 1), 'LineWidth', 1.5); % plotting the first column vector

    hold on;

    plot(x_ground_truth(:, 1), 'LineWidth', 1.5); % plotting the first column vector
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