function [x_reshaped,x_lasso_reshaped, gamma_reshaped, changeMap_reshaped] = compare(x_ground_truth)
%Compare.m This function processes a data image in both cJHBL and Lasso
%   Data y is converted into image by cJHBL and Lasso Algorithm along with
%   change detection

    
    assert(ndims(x_ground_truth) == 3, 'x_ground_truth must be a 3D array of size (n, n, J)');

    % x_ground_truth will be accepted as size (n, n, J)
    [n, ~, J] = size(x_ground_truth);

    % initialisations and hyperparameters
    max_iterations = 1e4;
    max_difference = 1e-3;
    
    eta_alpha = 1;
    eta_beta = eta_alpha;
    eta_gamma = 1;
    theta_alpha = 1e-3;
    theta_beta = theta_alpha;
    theta_gamma = theta_beta;
    
    noise_dimension = n*n;
    noise_mean = 0;
    noise_sd = 0.1;


    % data creation and other initialisations
    R = sparse_operator(n,n,1);
    K = size(R, 1);
    m = n;
    F = @(x) reshape(fft2(reshape(x,n,m))/sqrt(n*m),n*m,1);
    FH = @(x) reshape(ifft2(reshape(x,n,m))*sqrt(n*m),n*m,1);
    y = zeros(n^2, J);

    for jj = 1:J
        curr_truth_j = x_ground_truth(:, :, jj);
        noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
        y(:, jj) = F(curr_truth_j(:)) + noise;
    end

    x = randn(n^2, J) + 1i * randn(n^2, J);
    gamma = ones(J - 1, n^2);
    
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
    
    %% Visualizations - Complex LASSO
    
    x_lasso_reshaped = reshape(x_lasso_reconstruct(:, :), n, n, J);
    
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
    
    % -----------Jackowatz part---------------
    
    changeMap  = icd_batch(x_lasso_reconstruct, 5);        % 7×7 ML window (as used in the book's example)    
    changeMap_reshaped = reshape(changeMap(:,:), n, n, J-1);
    
    figchangeMap = figure('Name','ChangeMap plot Jackowatz','NumberTitle','off');
    for jj = 1:J-1
        subplot(2,J,jj);
        % imshow(changeMap_reshaped(:, :, jj), []);
        imagesc(changeMap_reshaped(:, :, jj));colormap(bone);colorbar;
        title(sprintf('Change map from img %d to %d',jj, jj+1));
    end
    % ----jackowatz ends-----------------------
    
    %% Visualizations - Complex JHBL
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

end