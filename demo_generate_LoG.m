clear;
clc;

function gaus_kernel_0 = generate_gaus_0(T)
    % parameters
    sigma = T/3;
    x = -T:T;

    % 1D Gaussian formula
    gaus_kernel_0 = exp(-x.^2 / (2*sigma^2));
    gaus_kernel_0 = gaus_kernel_0 ./ norm(gaus_kernel_0);
end

function gaus_kernel_1 = generate_gaus_1(T)
    % parameters
    sigma = T/3;
    x = -T:T;

    % 1D Gaussian formula
    gaus_kernel_0 = exp(-x.^2 / (2*sigma^2));
    gaus_kernel_1 = (-x / sigma^2) .* gaus_kernel_0;
    gaus_kernel_1 = gaus_kernel_1 ./ norm(gaus_kernel_1);
end

function gaus_kernel_2 = generate_gaus_2(T)
    % parameters
    sigma = T/3;
    x = -T:T;

    % 1D Laplacian of Gaussian formula
    gaus_kernel_0 = exp(-x.^2 / (2*sigma^2));
    gaus_kernel_2 = (x.^2/sigma^4 - 1/sigma^2) .* gaus_kernel_0;
    gaus_kernel_2 = gaus_kernel_2 ./ norm(gaus_kernel_2);
end

function noised = add_white_noise(mask, k)
    % generate a vector of white noise to add on data
    err = k * (rand(size(mask)) - 0.5);

    % add noise
    noised = data + err;
end

% DEBUG (T = 2)
T = 1;
impuls = .75;
noise_0 = generate_gaus_0(T);
noise_1 = generate_gaus_1(T);
noise_2 = generate_gaus_2(T);

% power analysis
fprintf('impulse: %d\ngaus_0: %d\ngaus_1: %d\ngaus_2: %d\n', impuls, std(noise_0), std(noise_1), std(noise_2));

% % plotting
% figure;
% subplot(3, 2, 1);
% plot(-T:T, noise_0, 'LineWidth', 2);
% title(['Gaussian function (\sigma = ' num2str(T/3) ')']);
% grid on;
% subplot(3, 2, 3);
% plot(-T:T, noise_1, 'LineWidth', 2);
% title(['First order derivative (\sigma = ' num2str(T/3) ')']);
% grid on;
% subplot(3, 2, 5);
% plot(-T:T, noise_2, 'LineWidth', 2);
% title(['Second order derivative (\sigma = ' num2str(T/3) ')']);
% grid on;

% % randomization
% k = 0.05;
% noised_0 = add_white_noise(noise_0, k);
% subplot(3, 2, 2);
% plot(-T:T, noised_0, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;
% noised_1 = add_white_noise(noise_1, k);
% subplot(3, 2, 4);
% plot(-T:T, noised_1, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;
% noised_2 = add_white_noise(noise_2, k);
% subplot(3, 2, 6);
% plot(-T:T, noised_2, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;

% % DEBUG (T = 7)
% T = 7;
% noise_0 = generate_gaus_0(T);
% noise_1 = generate_gaus_1(T);
% noise_2 = generate_gaus_2(T);

% % plotting
% figure
% subplot(3, 2, 1);
% plot(-T:T, noise_0, 'LineWidth', 2);
% title(['Gaussian function (\sigma = ' num2str(T/3) ')']);
% grid on;
% subplot(3, 2, 3);
% plot(-T:T, noise_1, 'LineWidth', 2);
% title(['First order derivative (\sigma = ' num2str(T/3) ')']);
% grid on;
% subplot(3, 2, 5);
% plot(-T:T, noise_2, 'LineWidth', 2);
% title(['Second order derivative (\sigma = ' num2str(T/3) ')']);
% grid on;

% % randomization
% noised_0 = add_white_noise(noise_0, k);
% subplot(3, 2, 2);
% plot(-T:T, noised_0, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;
% noised_1 = add_white_noise(noise_1, k);
% subplot(3, 2, 4);
% plot(-T:T, noised_1, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;
% noised_2 = add_white_noise(noise_2, k);
% subplot(3, 2, 6);
% plot(-T:T, noised_2, 'LineWidth', 2);
% title(['Gaussian with randomization (k = ' num2str(k) ')']);
% grid on;