clear; close all; clc;

% Linear-separable dataset
rng(0);
n = 100;
X_p = randn(n,2) + [2, -2];
X_n = randn(n,2) + [-1, -1];
X = [X_p; X_n];
y = [ones(n,1); -ones(n,1)];

C = 1.0;
linModel = svm_train(X, y, C, 'linear', struct(), 1e-3, 10);
[pred_lin, ~] = svm_predict(linModel, X);
acc_lin = mean(pred_lin == y);
figure('Name','SVM Linear Kernel');
plot_decision_views(X, y, linModel, sprintf("Linear"));

fprintf('Linear, C=%.2f, Acc=%.2f%%\n', C, acc_lin*100);

clear;

% Nonlinear dataset
rng(1);
n1 = 120; n2 = 120;
angles1 = 2*pi*rand(n1,1); r1 = 1 + 0.15*randn(n1,1); % inner circle (+1)
angles2 = 2*pi*rand(n2,1); r2 = 2 + 0.15*randn(n2,1); % outer circle (-1)
X1 = [r1.*cos(angles1), r1.*sin(angles1)];
X2 = [r2.*cos(angles2), r2.*sin(angles2)];
X = [X1; X2];
y = [ones(n1,1); -ones(n2,1)];

C = 1.0; sigma = 0.5;
rbfParam = struct('sigma', sigma);
rbfModel = svm_train(X, y, C, 'gaussian', rbfParam, 1e-3, 10);
[pred_rbf, ~] = svm_predict(rbfModel, X);
acc_rbf = mean(pred_rbf == y);
figure('Name','SVM Gaussian Kernel');
plot_decision_views(X, y, rbfModel,sprintf("Gaussian kernel"));

fprintf('Gaussian, C=%.2f, sigma=%.2f, Acc=%.2f%%\n', C, sigma, acc_rbf*100);
