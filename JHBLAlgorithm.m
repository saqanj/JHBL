%% JHBL (Joint Hierarchical Bayesian Learning) Algorithm
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023)
% By: Shamsher Tamang & Saqlain Anjum

%% Equation Parameters
J = 10; % Num. Images
n = 1; % Value of n for nXn Dimension
n_1D = n^2; % Flattened length for 1-D Case
max_iterations = 10^3; % For algorithm stopping condition.
max_difference = 10^-3; % For algorithm stopping condition.

%% Equation Hyperparameters
eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 2;

theta_alpha = 10^-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

%% Defining General Linear Transforms
R = eye(n); % Most general choice for R
K = n; % Number of rows in R

%% Defining the Forward Operator & Data
F = eye(n);  % Forward operators
y = zeros(J,1); % Measured data
x_ground_truth = zeros(J, n); % Ground truth images.
sparsity_level = 8; % number of non-zeros
noise_mean = 0;
noise_sd = 0.1;
noise_dimension = n;

idx_pool = randi([1, n], sparsity_level, 1);    % the indices to be changed to 1
for j = 1:J
    curr_truth_j = zeros(n, 1);
    curr_truth_j(idx_pool) = 1;
    x_ground_truth(j, :) = curr_truth_j;
    noise = noise_mean + noise_sd .* randn(noise_dimension, 1);   % noise of normal distribution with noise_mean and noise_sd
    y(j, :) = F * curr_truth_j + noise;
    idx_pool(end-1:end) = mod(idx_pool(end-1:end), n) + 1;    % update the last two indices to force changes, changes made here
end

%% Initialization (Algorithm Step 1)
x = randn(J, n); % J x 1 matrix, images to reconstruct
alpha = ones(J, 1); % J x 1 matrix, alpha hyperparameters, noise precision
beta = ones(J, K); % J x 1 matrix, beta hyperparameters, 
                  % intra-image regularization weight
gamma = ones(J - 1, n); % J-1 x 1 matrix, gamma hyperparameters, 
                       %  NOTE: gamma is a coupling weight between 
                       %  image x{j} and x{j+1}, so it must be a 
                       %  J-1 x 1 matrix.
max_iter = 10^3; % Maximum number of loop iterations                     
%% Algorithmic Iterations (All Remaining Steps)
M = zeros(J, 1);
for j = 1:J
    M(j) = length(y(j, :));
end

for l = 1:max_iter
    x_old = x;

    % -- alpha update
    for j = 1:J
        alpha(j, :) = (eta_alpha + M(j)/2 - 1) / (theta_alpha + 0.5 * norm(F * x(j, :) - y(j, :))^2);
    end

    % -- beta update
    for j = 1:J
        R_x = R * x(j); % k*1
        % beta(j,:) is 1*k for all k
        beta(j, :) = (eta_beta - 0.5) ./ (theta_beta + 0.5 * (R_x(:)').^2); % need to confirm what the [R_x]_k mean?
    end

    % -- gamma update
    for j = 2:J
        % x(j, :) --> 1*n, gamma(j, :) --> 1*n
        gamma(j-1, :) = (eta_gamma - 0.5) ./ (theta_gamma + 0.5 * (x(j-1, :) - x(j, :)).^2);
    end

    % -- x update
    % computing change masks
    
    for j = 1:J
        %computing G for each image
        % computing change mask for G and b
        if j == 1
            change_mask_for_G = diag(gamma(j, :));
            change_mask_term_for_b = diag(gamma(j, :)) * x(j+1, :);
        elseif j == J
            change_mask_for_G = diag(gamma(j-1, :));
            change_mask_term_for_b = diag(gamma(j-1, :)) * x(j-1, :);
        else
            change_mask_for_G = diag(gamma(j-1, :)) + diag(gamma(j, :));
            change_mask_term_for_b = diag(gamma(j-1, :)) * x(j-1, :) + diag(gamma(j, :)) * x(j+1, :);
        end
        
        % computing B_j
        B_j = diag(beta(j, :));

        G_j = alpha(j, :)*(F')*F+ (R')*B_j*R + change_mask_for_G;
        b_j = alpha(j, :)*(F')*y(j, :) + change_mask_term_for_b;
                
        % solving x from the linear system in equation 34
        %for solving we use gradient desncet iterating 5 times, citation 24

        % solution of x using citation 24
        r = b_j - G_j * x(j, :);
        for i = 1:5
            G_r = G_j * r;
            step_size = (r')*r/(r'*G_r);
            x(j, :) = x(j, :) + step_size * r;
            r = r - step_size * G_r;
        end
        % solution of x ended

    end

    % terminate loop if absolute and relative change condition is met
    % we compute the average absolute and relative change between two subsequent image
    % sequences
    absolute_change_matrix(:, :) = x(:,:) - x_old(:, :);
    absolute_change_norm_sum = 0;
    relative_change_norm_sum = 0;
    for i = 1:J
        abs_change_norm = norm(absolute_change_matrix(j, :), 2);
        absolute_change_norm_sum = absolute_change_norm_sum + abs_change_norm;
        relative_change_norm_sum = absolute_change_norm_sum + abs_change_norm/norm(x(j, :), 2);
    end
    average_absolute_change = absolute_change_norm_sum/(J-1);
    average_relative_change = relative_change_norm_sum/(J-1);

    if average_absolute_change < 10^(-3) && average_relative_change < 10^(-3)
        break
    end
end

%% Generating Visualizations
figure;
plot(x_ground_truth(j, :));
plot(x(j, :));