clear; close all; clc;

num_samples = 300;
num_classes = 3;
rng(42);

X1 = randn(num_samples/num_classes, 2) * 0.8 + [0, 1];
y1 = ones(size(X1, 1), 1);

X2 = randn(num_samples/num_classes, 2) * 0.8 + [-1, -1];
y2 = 2 * ones(size(X2, 1), 1);

X3 = randn(num_samples/num_classes, 2) * 0.8 + [1, -1];
y3 = 3 * ones(size(X3, 1), 1);

X = [X1; X2; X3];
y = [y1; y2; y3];

cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test = X(test_idx, :);
y_test = y(test_idx, :);


k_value = 5; 

fprintf('--- 运行基于 KD-Tree 的 k-NN 分类器 ---\n');
fprintf('训练集大小: %d | 测试集大小: %d | k: %d\n', ...
        size(X_train, 1), size(X_test, 1), k_value);

tic;
[final_accuracy, y_pred_test] = knn(X_train, y_train, X_test, y_test, k_value);
toc;

fprintf('\nAccuracy: %.4f\n', final_accuracy);