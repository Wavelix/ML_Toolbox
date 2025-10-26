clear; close all; clc;

% 生成数据
rng(42);
num_samples = 200;
w = [2]';
num_features = length(w);
X = 10 * rand(num_samples, num_features);
y = 1 + X * w + randn(num_samples, 1);

% X = [ones(m, 1), X_data]; 

lr = 0.0001;      
epochs = 10000;

% BGD
fprintf('--- 运行 BGD ---\n');
tic;
[theta_bgd, loss_bgd] = linearReg(X, y, lr, epochs, 'BGD', num_samples);
t_bgd = toc;
theta_str_bgd = strjoin(cellstr(sprintf('; %f', theta_bgd)), '');
fprintf('BGD 最终参数 theta: [%s]\n', theta_str_bgd(3:end));
fprintf('BGD 耗时: %f 秒\n', t_bgd);
fprintf('\n');

% MBGD
batch_size = 32;
fprintf('--- 运行 MBGD ---\n');
tic;
[theta_mbgd, loss_mbgd] = linearReg(X, y, lr, epochs, 'MBGD', batch_size);
t_mbgd = toc;
theta_str_mbgd = strjoin(cellstr(sprintf('; %f', theta_mbgd)), '');
fprintf('MBGD 最终参数 theta: [%s]\n', theta_str_mbgd(3:end)); 
fprintf('MBGD 耗时: %f 秒\n', t_mbgd);
fprintf('\n');

% SGD
lr = 0.0001;
epochs = 10000;
fprintf('--- 运行 SGD ---\n');
tic;
[theta_sgd, loss_sgd] = linearReg(X, y, lr, epochs, 'SGD', 1);
t_sgd = toc;
theta_str_sgd = strjoin(cellstr(sprintf('; %f', theta_sgd)), '');
fprintf('SGD 最终参数 theta: [%s]\n', theta_str_sgd(3:end)); 
fprintf('SGD 耗时: %f 秒\n', t_sgd);
fprintf('\n');


figure;
plot(1:length(loss_bgd), loss_bgd, 'LineWidth', 2, 'DisplayName', 'BGD Cost');
hold on;
plot(1:length(loss_sgd), loss_sgd, 'LineWidth', 2, 'DisplayName', 'SGD Cost');
plot(1:length(loss_mbgd), loss_mbgd, 'LineWidth', 2, 'DisplayName', ['MBGD Cost (Batch=', num2str(batch_size), ')']);
title('Loss Curve');
xlabel('epochs');
ylabel('Loss function');
legend;
grid on;

if num_features==1
    figure;
    x_plot = [min(X(:,1))-1, max(X(:,1))+1];
    y_plot = theta_mbgd(1) + theta_mbgd(2) * x_plot;
    scatter(X, y, 'o', 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none');
    hold on;
    plot(x_plot, y_plot, 'k-', 'LineWidth', 2);
    grid on;
end
