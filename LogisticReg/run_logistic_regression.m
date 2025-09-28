clear; close all; clc;

num_samples = 200;
num_features = 2;
rng(42);

X1 = randn(num_samples/2, num_features) * 2 - 2;
y1 = zeros(num_samples/2, 1);

X2 = randn(num_samples/2, num_features) * 2 + 2;
y2 = ones(num_samples/2, 1);

X = [X1; X2];
y = [y1; y2];

% 划分训练集和测试集
train_ratio = 0.8;
cv = cvpartition(size(X, 1), 'HoldOut', 1 - train_ratio);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

lr = 0.01;
epochs = 500;
method = 'mbgd';
batch_size = 32;
k_folds = 5;

fprintf('开始使用 %s 训练逻辑回归模型...\n', method);
tic;
[theta, train_losses, val_losses] = logisticReg(X_train, y_train, lr, epochs, method, batch_size, k_folds);
toc;
fprintf('训练完成。\n');
fprintf('训练得到的参数 theta:\n');
disp(theta);

figure;
plot(1:epochs, train_losses, 'b', 'LineWidth', 2);
hold on;
plot(1:epochs, val_losses, 'r', 'LineWidth', 2);
title('Loss Curve');
xlabel('Epochs');
ylabel('Loss');
legend('Training loss', 'Validation loss');
grid on;

if num_features == 2
    figure;
    
    % 绘制训练数据
    scatter(X_train(y_train == 0, 1), X_train(y_train == 0, 2), 'o', 'filled', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
    hold on;
    scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 's', 'filled', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');
    
    % 绘制测试数据
    scatter(X_test(y_test == 0, 1), X_test(y_test == 0, 2), 'o', 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
    scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 's', 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    
    % 绘制决策边界
    x_plot = [min(X(:,1))-1, max(X(:,1))+1];
    y_plot = (-1/theta(3)) * (theta(2) * x_plot + theta(1));
    plot(x_plot, y_plot, 'k-', 'LineWidth', 2);
    
    title('Decision boundary');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('training set - class 0', 'training set - class 1', 'test set - class 0', 'test set - class 1', 'decision boundary', 'Location', 'Best');
    grid on;
end