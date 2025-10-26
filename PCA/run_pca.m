clear; clc; close all;

centerData = true;      % mean-center features
standardize = true;     % z-score features

thisDir  = fileparts(mfilename('fullpath'));
dataPath = fullfile(thisDir, '..', 'wine.data');
if ~exist(dataPath, 'file')
    error('Data file not found: %s', dataPath);
end

raw = dlmread(dataPath, ',');

y = raw(:, 1);      
X = raw(:, 2:end);  
[n, d] = size(X);
fprintf('Loaded wine dataset: %d samples, %d features\n', n, d);

% ---- PCA to 2D ----
[k2_Z, k2_W, k2_mu, k2_evals, k2_explained, k2_Xrec, k2_mse, k2_rmse] = ...
    pca(X, 2, 'Center', centerData, 'Standardize', standardize);

fprintf('\nPCA to 2D:\n');
fprintf('  Explained variance (PC1, PC2): [%.2f%%, %.2f%%], cumulative: %.2f%%\n', ...
    100*k2_explained(1), 100*k2_explained(2), 100*sum(k2_explained));
fprintf('  Reconstruction MSE:  %.6f\n', k2_mse);
fprintf('  Reconstruction RMSE: %.6f\n', k2_rmse);

% Plot 2D
figure('Name','PCA 2D'); hold on; grid on; box on;
cls = unique(y);
cols = lines(numel(cls));
for i = 1:numel(cls)
    idx = (y == cls(i));
    scatter(k2_Z(idx,1), k2_Z(idx,2), 35, cols(i,:), 'filled');
end
xlabel('PC1'); ylabel('PC2');
title(sprintf('PCA to 2D'));
legend();
axis equal; hold off;

% ---- PCA to 3D ----
[k3_Z, k3_W, k3_mu, k3_evals, k3_explained, k3_Xrec, k3_mse, k3_rmse] = ...
    pca(X, 3, 'Center', centerData, 'Standardize', standardize);

fprintf('\nPCA to 3D:\n');
fprintf('  Explained variance (PC1, PC2, PC3): [%.2f%%, %.2f%%, %.2f%%], cumulative: %.2f%%\n', ...
    100*k3_explained(1), 100*k3_explained(2), 100*k3_explained(3), 100*sum(k3_explained));
fprintf('  Reconstruction MSE:  %.6f\n', k3_mse);
fprintf('  Reconstruction RMSE: %.6f\n', k3_rmse);

% Plot 3D
figure('Name','PCA 3D'); hold on; grid on; box on;
for i = 1:numel(cls)
    idx = (y == cls(i));
    scatter3(k3_Z(idx,1), k3_Z(idx,2), k3_Z(idx,3), 35, cols(i,:), 'filled');
end
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title(sprintf('PCA to 3D'));
legend();
axis vis3d; view(45, 25); hold off;
