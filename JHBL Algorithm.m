%% JHBL (Joint Hierarchical Bayesian Learning) Algorithm
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023)
% By: Shamsher Tamang & Saqlain Anjum

%% Equation Parameters
J = 10; % Num. Images
n = 128; % Value of n for nXn Dimension
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
R = speye(n); % Most general choice for R
K = n; % Number of rows in R

%% Defining the Forward Operator & Data
F = cell(J,1); % Forward operators
y = cell(J,1); % Measured data
x_ground_truth = cell(J, 1); % Ground truth images.

for j = 1:J
    curr_truth_j = randn(n, 1);
    x_ground_truth{j} = curr_truth_j;
    F_j = eye(n);
    F{j} = F_j;
    F_j_rows = n;
    noise = sqrt(1 / eta_alpha) * randn(F_j_rows, 1); % Equation 5 in Paper
    y{j} = F_j * curr_truth_j + noise;
end

%% Initialization (Algorithm Step 1)
x = cell(J, 1); % J x 1 matrix, images to reconstruct
alpha = ones(J,1); % J x 1 matrix, alpha hyperparameters, noise precision
beta = ones(J,1); % J x 1 matrix, beta hyperparameters, 
                  % intra-image regularization weight
gamma = ones(J - 1,1); % J-1 x 1 matrix, gamma hyperparameters, 
                       %  NOTE: gamma is a coupling weight between 
                       %  image x{j} and x{j+1}, so it must be a 
                       %  J-1 x 1 matrix.
for j=1:J
    x{j} = randn(n, 1);
    beta{j} = ones(K, 1);
end
for j=1:J-1
    gamma{j} = ones(n, 1);
end
%% Algorithmic Iterations (All Remaining Steps)
for l = 1:max_iter
    x_old = x;

    % -- alpha update
    for j = 1:J
        M_j = size(F{j},1)
        alpha{j} = (eta_alpha + M_j/2 - 1) / (theta_alpha + 0.5 * norm(F{j} * x{j} - y{j})^2);
    end

    % -- beta update
    for j = 1:J
        R_x = R * x{j};
        beta{j} = (eta_beta - 0.5) ./ (theta_beta + 0.5 * R_x.^2);
    end

    % -- gamma update
    for j = 2:J
        gamma{j-1} = (eta_gamma - 0.5) ./ (theta_gamma + 0.5 * (x{j-1} - x{j}).^2);
    end

    % change mask code: 
    % change mask entry = 0 if change region
    % change mask entry = 1 if no change region
    % equation 3 ****, ask about tomorrow if make no sense

    % also add norm difference code down here, it is correct.
