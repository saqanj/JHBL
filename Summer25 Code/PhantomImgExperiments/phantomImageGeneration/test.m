J = 3;
n = 50;

[mag, phase, II] = phantom_image_gen('blockySinosudol', J, n, 18, 25, 10^(-3), 10^(-3), 10^(-3), 0.25, 0);


figure('Name','Synthetic SAR sequence','NumberTitle','off');
for jj = 1:J
    subplot(2,J,jj);
    imshow(reshape(mag(:, jj), [n,n]),[]); 
    title(sprintf('Mag %d',jj));

    subplot(2,J,J+jj);
    imagesc( reshape(phase(:,jj), [n, n]));
    axis image off;
    colormap(gray);
    colorbar;
    title(sprintf('Phase %d',jj));
end