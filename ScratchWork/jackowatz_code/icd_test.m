% icd_test.m
rng(4)

tic

%% Parameters
J = 3;
n = 100;

m = n;
F = @(x) reshape(fft2(reshape(x,n,m))/sqrt(n*m),n*m,1);
FH = @(x) reshape(ifft2(reshape(x,n,m))*sqrt(n*m),n*m,1);
y = zeros(n^2, J);
x_ground_truth = complex(zeros(n, m, J));
noise_dimension = n*m;

sparsity_level = 8;
noise_mean = 0;
noise_sd = 0.1;

mag = zeros(n);
mag(10:50, 10:50)   = randi([0, 2]);
mag(50:100, 10:50)  = randi([0, 2]);
mag(10:50, 60:100)  = randi([0, 2]);
mag(16:25, 6:25) = randi([0, 2]);

mag = mag + 0.5;

for j = 1:J
    [X, Y] = meshgrid(1:n, 1:n);
    phase = (pi/10 * j) * sin(2*pi*X/n) .* cos(2*pi*Y/n);
    curr_truth_j = mag .* exp(1i * phase);
    x_ground_truth(:, :, j) = curr_truth_j;
    noise = noise_mean + noise_sd/sqrt(2) * (randn(noise_dimension, 1)+1i*randn(noise_dimension,1));
    y(:, j) = F(curr_truth_j(:)) + noise;
end

y_img = zeros(size(y));
for j = 1:J
    y_img(:, j) = FH(y(:, j));
end


% ---------------------------------
% calling changeMap code (Jackowatz)
changeMap  = icd_batch(y_img, 7);        % 7×7 ML window (as used in the book's example)
% ---------------------------------

changeMap_reshaped = reshape(changeMap(:,:), n, n, J-1);
reconstruction_reshaped = x_ground_truth(:, :, :);

figMag = figure('Name','ChangeMap plot Jackowatz','NumberTitle','off');
for jj = 1:J-1
    subplot(2,J,jj);
    imshow(changeMap_reshaped(:, :, jj), []);
    title(sprintf('Change map from img %d to %d',jj, jj+1));
end
for jj = 1:J
    subplot(2,J,J+jj);
    imshow(abs(reconstruction_reshaped(:, :, jj)), []);
    title(sprintf('Reconstruction img %d',jj));
end
toc