clear all;
rng(4)

tic
%% Complex-Valued JHBL Comparison with Complex LASSO on Coherent Change Detection
% By: Saqlain Anjum & Shamsher Ghising Tamang

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

real_mag = mag + 0.5;

for j = 1:J
    if is_2D
        mag = real_mag;
        magChange = 1.5;
        r = randi(n-8); % row
        c = randi(n-8); % column
        mag(r:r+8-1, c:c+8-1) = mag(r:r+8-1, c:c+8-1)+ magChange;
        [X, Y] = meshgrid(1:n, 1:n);
        % phase = ones(n, n) * pi/4;
        % phase(31:40, :) = pi/2 + 0.1*1; % First tophat region 
        % phase(41:50, :) = pi/2 + 0.1*2; % Second tophat region 
        % phase(51:60, :) = pi/2 + 0.1*3; % Third tophat region 
        phase = (pi/2) * sin(2*pi*X/n) .* cos(2*pi*Y/n);
        r = randi(n-8); % row
        c = randi(n-10); % column
        phase(r:r+8-1, c:c+10-1) = magChange;

        curr_truth_j = mag .* exp(1i * phase);
        x_ground_truth(:, :, j) = curr_truth_j;
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
        y(:, j) = F(curr_truth_j(:)) + noise;
    else
        mag = [ones(30,1); 2*ones(30+j,1); 1.5*ones(40-j,1)];
        phase = ones(n, 1) * pi/4; 
        phase(31:40) = pi/2 + 0.1*1; % First tophat region
        phase(41:50) = pi/2 + 0.1*2; % Second tophat region
        phase(51:60) = pi/2 + 0.1*3; % Third tophat region
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

%% Complex LASSO Regeneration
for l = 1:J
    x_lasso_reconstruct(:, l) = generalized_complex_lasso([], y(:, l), R, 0.1, 0.1, FH, 0);
end
%% Visualizations - Ground Truth
if is_2D
    gt_reshaped = x_ground_truth(:, :, :);
    
    figGroundTruth = figure('Name','GroundTruth','NumberTitle','off');
    for jj = 1:J
        subplot(2,J,jj);
        imshow(abs(gt_reshaped(:, :, jj)), []);
        title(sprintf('Act. Mag %d',jj));
        
        subplot(2,J,J+jj);
        imshow(angle(x_lasso_reshaped(:, :, jj)), []);
        title(sprintf('Act. Phase %d',jj));
    end
else
    % For 1-D Data: Plot magnitude and phase
    figure;
    subplot(2,1,1); 
    plot(abs(x_lasso_reconstruct(:, 1)), 'LineWidth', 1.5); 
    hold on; 
    plot(abs(x_ground_truth(:, 1)), '--', 'LineWidth', 1.5);
    hold off;
    legend('Reconstructed Magnitude','Ground Truth Magnitude');
    title('1D Magnitude Comparison');
    
    subplot(2,1,2); 
    plot(angle(x_lasso_reconstruct(:, 1)), 'LineWidth', 1.5); 
    hold on; 
    plot(angle(x_ground_truth(:, 1)), '--', 'LineWidth', 1.5);
    hold off;
    legend('Reconstructed Phase','Ground Truth Phase');
    title('1D Phase Comparison');
end

%% Visualizations - Complex LASSO
if is_2D
  % Reshape the reconstructed signal for Complex LASSO
    x_lasso_reshaped = reshape(x_lasso_reconstruct(:, :), n, n, J);
    % gt_reshaped = x_ground_truth(:, :, 1);

    % % Plot reconstructed magnitude and phase for Complex LASSO
    % figure; imshow(abs(x_lasso_reshaped), []); title('Reconstructed Magnitude (Lasso)'); 
    % figure; imshow(angle(x_lasso_reshaped), []); title('Reconstructed Phase (Lasso)'); 
    % 
    % % Plot ground truth magnitude and phase
    % figure; imshow(abs(gt_reshaped), []); title('Ground Truth Magnitude (Image 1)'); 
    % figure; imshow(angle(gt_reshaped), []); title('Ground Truth Phase (Image 1)');

    % gt_reshaped = x_ground_truth(:, :, :);
    
    figLasso = figure('Name','Lasso Reconstruction','NumberTitle','off');
    for jj = 1:J
        subplot(2,J,jj);
        imshow(abs(x_lasso_reshaped(:, :, jj)), []);
        title(sprintf('Pred. Mag (Lasso) %d',jj));
        
        subplot(2,J,J+jj);
        imshow(angle(x_lasso_reshaped(:, :, jj)), []);
        title(sprintf('Pred. Phase (Lasso) %d',jj));
    end
else
    % For 1-D Data: Plot magnitude and phase
    figure;
    subplot(2,1,1); 
    plot(abs(x_lasso_reconstruct(:, 1)), 'LineWidth', 1.5); 
    hold on; 
    plot(abs(x_ground_truth(:, 1)), '--', 'LineWidth', 1.5);
    hold off;
    legend('Reconstructed Magnitude','Ground Truth Magnitude');
    title('1D Magnitude Comparison');
    
    subplot(2,1,2); 
    plot(angle(x_lasso_reconstruct(:, 1)), 'LineWidth', 1.5); 
    hold on; 
    plot(angle(x_ground_truth(:, 1)), '--', 'LineWidth', 1.5);
    hold off;
    legend('Reconstructed Phase','Ground Truth Phase');
    title('1D Phase Comparison');
end

% -----------Jackowatz part---------------
if is_2D
    changeMap  = icd_batch(x_lasso_reconstruct, 7);        % 7×7 ML window (as used in the book's example)
    
    changeMap_reshaped = reshape(changeMap(:,:), n, n, J-1);
    reconstruction_reshaped = reshape(x_lasso_reconstruct(:, :), n, n, J);
    
    figchangeMap = figure('Name','ChangeMap plot Jackowatz','NumberTitle','off');
    for jj = 1:J-1
        subplot(2,J,jj);
        imshow(changeMap_reshaped(:, :, jj), []);
        title(sprintf('Change map from img %d to %d',jj, jj+1));
    end
    % for jj = 1:J
    %     subplot(2,J,J+jj);
    %     imshow(abs(reconstruction_reshaped(:, :, jj)), []);
    %     title(sprintf('Sample Reconstruction img %d',jj));
    % end
end
% ----jackowatz ends-----------------------

%% Visualizations - Complex JHBL
if is_2D
    gt_reshaped = x_ground_truth(:, :, :);
    
    figActual = figure('Name','Mag Phase Actual','NumberTitle','off');
    for jj = 1:J
        subplot(2,J,jj);
        imshow(abs(gt_reshaped(:, :, jj)), []);
        title(sprintf('Act. Mag %d',jj));
        
        subplot(2,J,J+jj);
        imshow(angle(gt_reshaped(:, :, jj)), []);
        title(sprintf('Act. Phase %d',jj));
    end

    % x_reshaped = reshape(x(:, 1), n, n);
    % gt_reshaped = x_ground_truth(:, :, 1);
    % 
    % figure; imshow(abs(x_reshaped), []); title('Reconstructed Magnitude (Image 1)');
    % figure; imshow(angle(x_reshaped), []); title('Reconstructed Phase (Image 1)');
    % figure; imshow(abs(gt_reshaped), []); title('Ground Truth Magnitude (Image 1)');
    % figure; imshow(angle(gt_reshaped), []); title('Ground Truth Phase (Image 1)');
else
    figure;
    subplot(2,1,1); plot(abs(x(:,1)), 'LineWidth', 1.5); hold on; plot(abs(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
    legend('Reconstructed Magnitude','Ground Truth Magnitude');
    title('1D Magnitude Comparison');

    subplot(2,1,2); plot(angle(x(:,1)), 'LineWidth', 1.5); hold on; plot(angle(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
    legend('Reconstructed Phase','Ground Truth Phase');
    title('1D Phase Comparison');
end


% ------gamma plots-------------------
gamma_reshaped = reshape(gamma', n, n, J-1);
figPhase = figure('Name','Gamma','NumberTitle','off');
for jj = 1:J-1
    normalised_gamma = gamma_reshaped(:, :, jj)/max(gamma_reshaped(:, :, jj), [], "all");
    subplot(2,J,jj);
    imagesc(normalised_gamma(:, :));
    % axis image off;
    colormap(gray);
    colorbar;
    title(sprintf('Gamma Img %d to %d',jj, jj+1));
end
toc