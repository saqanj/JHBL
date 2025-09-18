import phantomImageGeneration.*

clear all;
rng(4)

tic
%%% Complex-Valued JHBL (Joint Hierarchical Bayesian Learning) Algorithm
%% Experiment on Phantom Images
% Based on Sequential Image Recovery Using Joint Hierarchical Bayesian
% Learning (Xiao & Glaubitz 2023) and Complex-Valued Signal Recovery Using
% a Generalized Bayesian LASSO (Green, Lindbloom, & Gelb 2025)
% By: Shamsher Tamang & Saqlain Anjum

%% Parameters
J = 3;
n = 150;
max_iterations = 1e4;
max_difference = 1e-3;
sparsity_level = 8;
noise_mean = 0;
noise_sd = 0.1;
phaseType = 'blockySinosudol';  % type of phase choose between 'sinosudol' and 'smooth'
deltaTheta = 18;          % rotation change of the ellipses between images
sigma = 25;                % tuning parameter for smooth phase, higher = bettter smooth phase; set to 25 for sinosudol
epsMag = 10^(-3);          % small error to make mag positively biased
epsPhase = 0;              % error added in each phase
epsPhase_variation = 10^(-3); % phase difference between subsequent images
phase_indices_to_change = 0.25; % percerntage of phase indices to change [0, 1]
visualise = 0;            % set to 1 to visualise the input generation

eta_alpha = 1;
eta_beta = eta_alpha;
eta_gamma = 2;
theta_alpha = 1e-3;
theta_beta = theta_alpha;
theta_gamma = theta_beta;

R = sparse_operator(n,n,1);
K = size(R, 1);
m = n;
F = @(x) reshape(fft2(reshape(x,n,m))/sqrt(n*m),n*m,1);
FH = @(x) reshape(ifft2(reshape(x,n,m))*sqrt(n*m),n*m,1);
y = zeros(n^2, J);
[mag, phase, x_ground_truth] = phantom_image_gen(phaseType, J, n, deltaTheta, sigma, epsMag, epsPhase, epsPhase_variation, phase_indices_to_change, visualise);
noise_dimension = n*m;

for j = 1:J
    curr_truth_j = x_ground_truth(:, j);
    noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
    y(:, j) = F(curr_truth_j(:)) + noise;
end

%% Initialization
x = randn(n^2, J) + 1i * randn(n^2, J);
gamma = ones(J - 1, n^2);

alpha = ones(J, 1);
beta = ones(J, K);
M = repmat(noise_dimension, J, 1);

%% JHBL Iterations
for l = 1:max_iterations
    x_old = x;

    for j = 1:J
        alpha(j) = (eta_alpha + M(j) - 1) / (theta_alpha + norm(abs(F(x(:, j)) - y(:, j)),2)^2);
    end

    %LTheta = zeros([size(R) J]);
    % for jj = 1:J
    %     Theta_jj = x(:, jj) ./ (abs(x(:, jj))); % phase calculation
    %     % apply TV only to magnitude
    %     LTheta(:, :, jj) = R .* Theta_jj';          
    % 
    %     R_x = LTheta(:, :, jj) * x(:, jj);
    %     beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);
    % end

    for j = 2:J
        gamma(j-1, :) = eta_gamma ./ (theta_gamma + abs(x(:, j-1) - x(:, j)).^2);
    end

    for j = 1:J
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
        
        Theta_j = x(:, j) ./ (abs(x(:, j))); % phase calculation
        % apply TV only to magnitude
        LTheta_j = R .* Theta_j';
        R_x = LTheta_j * x(:, j);
        beta(j, :) = eta_beta ./ (theta_beta + abs(R_x).^2);

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

%% Visualizations and Saving Results
projectRoot = fileparts(mfilename('fullpath'));
resultsRoot = fullfile(projectRoot,'Results');

if ~exist(resultsRoot,'dir')
    mkdir(resultsRoot)
end

ts = datestr(now,'yyyy-mm-dd_HH-MM');
resultsDir = fullfile(resultsRoot, ts);
mkdir(resultsDir);                                          % results/yyyy‑mm‑dd_HH‑MM
disp(['Saving figures to: ' resultsDir])

x_reshaped = reshape(x(:, :), n, n, J);
gt_reshaped = reshape(x_ground_truth(:, :), n, n, J);

figMag = figure('Name','Phantom Mag | Actual vs Prediction','NumberTitle','off');
for jj = 1:J
    subplot(2,J,jj);
    imshow(abs(gt_reshaped(:, :, jj)));
    title(sprintf('Act. Mag %d',jj));
    
    subplot(2,J,J+jj);
    imshow(abs(x_reshaped(:, :, jj)));
    title(sprintf('Pred. Mag %d',jj));
end

saveas(figMag, fullfile(resultsDir,'Phantom_Mag_Actual_vs_Prediction.png'));   % or .fig / .pdf
% close(figMag)

figPhase = figure('Name','Phantom Phase Actual vs Prediction','NumberTitle','off');
for jj = 1:J
    subplot(2,J,jj);
    imagesc(angle(gt_reshaped(:, :, jj)));
    axis image off;
    colormap(gray);
    colorbar;
    title(sprintf('Act. Phase %d',jj));
    
    subplot(2,J,J+jj);
    imagesc(angle(x_reshaped(:, :, jj)));
    axis image off;
    colormap(gray);
    colorbar;
    title(sprintf('Pred. Phase %d',jj));
end

saveas(figPhase, fullfile(resultsDir,'Phantom_Phase_Actual_vs_Prediction.png'));
% close(figPhase)

toc


%% ----------  Write README.md with experiment metadata  ----------
readmePath = fullfile(resultsDir,'README.md');
fid = fopen(readmePath,'w');

fprintf(fid,"# Experiment run: %s\n\n", ts);
fprintf(fid,"## Phantom‑image parameters\n");
fprintf(fid,"| Parameter | Value |\n");
fprintf(fid,"|-----------|-------|\n");
fprintf(fid,"| `J` (number of images) | %d |\n", J);
fprintf(fid,"| `n` (resolution)       | %d |\n", n);
fprintf(fid,"| Eta Alpha              | %d |\n", eta_alpha);
fprintf(fid,"| Eta Beta               | %d |\n", eta_beta);
fprintf(fid,"| Eta Gamma              | %d |\n", eta_gamma);
fprintf(fid,"| Theta Alpha            | %d |\n", theta_alpha);
fprintf(fid,"| Theta Beta             | %d |\n", theta_beta);
fprintf(fid,"| Theta Gamma            | %d |\n", theta_gamma);
fprintf(fid,"| Sparsity Level         | %d |\n", sparsity_level);
fprintf(fid,"| noise mean             | %d |\n", noise_mean);
fprintf(fid,"| noise_sd               | %d |\n", noise_sd);
fprintf(fid,"| R_operator (fill yourself)  |    |\n");
fprintf(fid,"| F operator (fill yourself)  |    |\n");

fprintf(fid,"\n## Output\n");
fprintf(fid,"* **Phantom_Mag_Actual_vs_Prediction.png** – magnitude comparison\n");
fprintf(fid,"* **Phantom_Phase_Actual_vs_Prediction.png** – wrapped‑phase comparison\n");

fprintf(fid,"\n---\nCreated automatically by *Experiment_Script.m* on %s.\n", datestr(now));

fclose(fid);
disp(['README.md created at: ' readmePath]);