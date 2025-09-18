clear all;
close all
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
n = 256;
max_iterations = 1e4;
max_difference = 1e-3;

eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 1;
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

% mag(10:50, 10:50)   = randi([0, 2]);
% mag(50:100, 10:50)  = randi([0, 2]);
% mag(10:50, 60:100)  = randi([0, 2]);
% mag(16:25, 6:25) = randi([0, 2]);

mag = phantom(n);
% [mag, ~, ~] = phantom_image_gen('smoothPhase', J, n, 18, 25, 10^(-3), 10^(-3), 10^(-3), 0.25, 0);

real_mag = mag + 0.5;

phase_gt = wrapToPi((rands([n,n],10)-.5)*pi);%(pi/2) * sin(20*pi*X/n) .* cos(20*pi*Y/n);
for jj = 1:J
    if is_2D
        mag = real_mag;
        magChange = 1.5;
        r = round(n/3); % row
        c = round(n/3) + 45*jj; % column
        mag(r:r+30-1, c:c+18-1) = magChange;
        [X, Y] = meshgrid(1:n, 1:n);
        % phase = ones(n, n) * pi/4;
        % phase(31:40, :) = pi/2 + 0.1*1; % First tophat region 
        % phase(41:50, :) = pi/2 + 0.1*2; % Second tophat region 
        % phase(51:60, :) = pi/2 + 0.1*3; % Third tophat region 
        
        r = round(2*n/3); % row
        c = round(n/3) + 45*jj; % column
        phase = phase_gt;
        phase(r:r+18-1, c:c+30-1) = magChange;

        curr_truth_j = mag .* exp(1i * phase);
        x_ground_truth(:, :, jj) = curr_truth_j;
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
        y(:, jj) = F(curr_truth_j(:)) + noise;
    else
        mag = [ones(30,1); 2*ones(30+jj,1); 1.5*ones(40-jj,1)];
        phase = ones(n, 1) * pi/4; 
        phase(31:40) = pi/2 + 0.1*1; % First tophat region
        phase(41:50) = pi/2 + 0.1*2; % Second tophat region
        phase(51:60) = pi/2 + 0.1*3; % Third tophat region
        x_ground_truth(:, jj) = mag .* exp(1i * phase);
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension, 1));
        y(:, jj) = F(x_ground_truth(:, jj)) + noise;
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
for ll = 1:max_iterations
    x_old = x;

    for jj = 1:J
        alpha(jj) = (eta_alpha + M(jj) - 1) / (theta_alpha + norm(abs(F(x(:, jj)) - y(:, jj)),2)^2);
    end

    % for j = 1:J
    %     Theta_j = x(:, j) ./ (abs(x(:, j))); % phase calculation
    %     % apply TV only to magnitude
    %     LTheta_j = R .* Theta_j';          
    % 
    %     R_x = LTheta_j * x(:, j);
    %     beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);
    % end

    for jj = 2:J
        gamma(jj-1, :) = eta_gamma ./ (theta_gamma + abs(x(:, jj-1) - x(:, jj)).^2);
    end

    for jj = 1:J
        Theta_j = x(:, jj) ./ (abs(x(:, jj))); % phase calculation
        % apply TV only to magnitude
        LTheta_j = R .* Theta_j';
        R_x = LTheta_j * x(:, jj);
        beta(jj, :) = eta_beta ./ (theta_beta + abs(R_x).^2);

        if jj == 1
            change_mask_G = gamma(jj, :).';
            change_term_b = gamma(jj, :).' .* x(:, jj+1);
        elseif jj == J
            change_mask_G = gamma(jj-1, :).';
            change_term_b = gamma(jj-1, :)' .* x(:, jj-1);
        else
            change_mask_G = gamma(jj-1, :).' + gamma(jj, :).';
            change_term_b = gamma(jj-1, :).' .* x(:, jj-1) + gamma(jj, :).'.* x(:, jj+1);
        end

        G_j = @(x_var) alpha(jj)*FH(F(x_var)) + LTheta_j' * (beta(jj,:).' .* (LTheta_j * x_var)) + change_mask_G .* x_var;
        b_j = alpha(jj)*FH(y(:, jj)) + change_term_b;

        r = b_j - G_j(x(:, jj));
        for i = 1:5
            G_r = G_j(r);
            step_size = (r') * r / (r' * G_r);
            x(:, jj) = x(:, jj) + step_size * r;
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
for ll = 1:J
    x_lasso_reconstruct(:, ll) = generalized_complex_lasso([], y(:, ll), R, 0.1, 0.1, FH, 0);
end
%% Visualizations - Ground Truth
if is_2D
    x_lasso_reshaped = reshape(x_lasso_reconstruct(:, :), n, n, J);
    gt_reshaped = x_ground_truth(:, :, :);
    
    figGroundTruth = figure('Name','GroundTruth','NumberTitle','off');
    for jj = 1:J
        subplot(2,J,jj);
        % imshow(abs(gt_reshaped(:, :, jj)), []);
        imagesc(abs(gt_reshaped(:, :, jj)));colormap(bone);colorbar;
        title(sprintf('Act. Mag %d',jj));
        
        subplot(2,J,J+jj);
        % imshow(angle(gt_reshaped(:, :, jj)), []);
        imagesc(angle(gt_reshaped(:, :, jj)));colormap(bone);colorbar;
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
        imagesc(abs(x_lasso_reshaped(:, :, jj)));colormap(bone);colorbar;
        % imshow(abs(x_lasso_reshaped(:, :, jj)), []);
        title(sprintf('Pred. Mag (Lasso) %d',jj));
        
        subplot(2,J,J+jj);
        % imshow(angle(x_lasso_reshaped(:, :, jj)), []);
        imagesc(angle(x_lasso_reshaped(:, :, jj)));colormap(bone);colorbar;
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
    changeMap  = icd_batch(x_lasso_reconstruct, 5);        % 7×7 ML window (as used in the book's example)
    
    changeMap_reshaped = reshape(changeMap(:,:), n, n, J-1);
    reconstruction_reshaped = reshape(x_lasso_reconstruct(:, :), n, n, J);
    
    figchangeMap = figure('Name','ChangeMap plot Jackowatz','NumberTitle','off');
    for jj = 1:J-1
        subplot(2,J,jj);
        % imshow(changeMap_reshaped(:, :, jj), []);
        imagesc(changeMap_reshaped(:, :, jj));colormap(bone);colorbar;
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
    x_reshaped = reshape(x(:, :), n, n, J);
    
    figActual = figure('Name','Mag Phase Prediction','NumberTitle','off');
    for jj = 1:J
        subplot(2,J,jj);
        imagesc(abs(x_reshaped(:, :, jj)));colormap(bone);colorbar;
        title(sprintf('Pred. Mag %d',jj));
        
        subplot(2,J,J+jj);
        imagesc(angle(x_reshaped(:, :, jj)));colormap(bone);colorbar;
        title(sprintf('Pred. Phase %d',jj));
    end
    set(gcf,'Position',[100,100,1000,500])

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

%% original noisy images
% if is_2D
%     gt_reshaped = x_ground_truth(:, :, :);
% 
%     figActual = figure('Name','Mag Phase Actual','NumberTitle','off');
%     for jj = 1:J
%         subplot(2,J,jj);
%         imshow(abs(gt_reshaped(:, :, jj)), []);
%         title(sprintf('Act. Mag %d',jj));
% 
%         subplot(2,J,J+jj);
%         imshow(angle(gt_reshaped(:, :, jj)), []);
%         title(sprintf('Act. Phase %d',jj));
%     end
% 
%     x_reshaped = reshape(FH(y(:, 1)), n, n);
%     gt_reshaped = x_ground_truth(:, :, 1);
% 
%     figure; imshow(abs(x_reshaped), []); title('Noisy Magnitude (Image 1)');
%     figure; imshow(angle(x_reshaped), []); title('Noisy Phase (Image 1)');
% else
%     figure;
%     subplot(2,1,1); plot(abs(FH(y(:, 1))), 'LineWidth', 1.5); hold on; plot(abs(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
%     legend('Noisy Magnitude','Ground Truth Magnitude');
%     title('1D Magnitude Comparison');
% 
%     subplot(2,1,2); plot(angle(FH(y(:, 1))), 'LineWidth', 1.5); hold on; plot(angle(x_ground_truth(:,1)), '--', 'LineWidth', 1.5);
%     legend('Noisy Phase','Ground Truth Phase');
%     title('1D Phase Comparison');
% end

toc