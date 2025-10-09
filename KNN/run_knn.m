clear; close all; clc;

num_samples = 300;
num_classes = 3;
rng(1);

X1 = randn(num_samples/num_classes, 2) * 0.5 + [0, 1];
y1 = ones(size(X1, 1), 1);

X2 = randn(num_samples/num_classes, 2) * 0.5 + [-1, -1];
y2 = 2 * ones(size(X2, 1), 1);

X3 = randn(num_samples/num_classes, 2) * 0.5 + [1, -1];
y3 = 3 * ones(size(X3, 1), 1);

X = [X1; X2; X3];
y = [y1; y2; y3];

% 划分训练集和测试集 (7:3)
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test = X(test_idx, :);
y_test = y(test_idx, :);

k_value = 5; 

fprintf('--- 运行 k-NN 分类器 ---\n');
fprintf('训练集大小: %d | 测试集大小: %d | 近邻数 k: %d\n', ...
        size(X_train, 1), size(X_test, 1), k_value);

tic;
accuracy = knn(X_train, y_train, X_test, y_test, k_value);
toc;

fprintf('最终测试集准确率为: %.4f\n', accuracy);

figure;
gscatter(X_test(:, 1), X_test(:, 2), y_test);
title(['k-NN 测试数据分布 (k = ', num2str(k_value), ')']);
xlabel('特征 1');
ylabel('特征 2');
legend('类别 1', '类别 2', '类别 3', 'Location', 'best');
grid on;