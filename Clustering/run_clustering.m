clear; clc; close all;

% 2D dataset
% k = 3; 
% n_samples = 150;
% centers = [2 2; 8 3; 5 8];
% X = [];
% for i = 1:k
%     X = [X; bsxfun(@plus, randn(n_samples, 2), centers(i, :))];
% end

% 3D dataset
k = 4;
n_samples = 100;
centers = [2 2 2; 8 8 2; 2 8 8; 8 2 8];
X = [];
for i = 1:k
    X = [X; bsxfun(@plus, randn(n_samples, 3), centers(i, :))];
end

[m, n] = size(X);
max_iters = 100;

% --- K-Means ---
% disp('Running K-Means...');
% [centroids, labels] = k_means(X, k, max_iters);
% title_text = 'K-Means Clustering';

% --- K-Means++ ---
disp('Running K-Means++...');
[centroids, labels] = k_means_pp(X, k, max_iters);
title_text = 'K-Means++ Clustering';

% --- Soft K-Means ---
% disp('Running Soft K-Means...');
% beta = 1; % Stiffness parameter
% [centroids, responsibilities] = soft_k_means(X, k, max_iters, beta);
% [~, labels] = max(responsibilities, [], 2); % Get hard assignments for plotting
% title_text = 'Soft K-Means Clustering';


if n == 2
    figure;
    hold on;
    gscatter(X(:,1), X(:,2), labels);
    plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
    title(title_text);
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');
    hold off;
elseif n == 3
    figure;
    hold on;
    scatter3(X(:,1), X(:,2), X(:,3), 36, labels, 'filled');
    plot3(centroids(:,1), centroids(:,2), centroids(:,3), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
    title(title_text);
    xlabel('Feature 1');
    ylabel('Feature 2');
    zlabel('Feature 3');
    view(3);
    grid on;
    hold off;
else
    disp('Data is not 2D or 3D, skipping plot.');
end

disp('Clustering complete.');
disp('Final centroids:');
disp(centroids);
