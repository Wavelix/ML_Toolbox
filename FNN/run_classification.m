clear; close all; clc;

num_samples = 600;
num_classes = 3;
rng(42);

X1 = randn(num_samples/num_classes, 2) * 0.8 + [0, 2];
X2 = randn(num_samples/num_classes, 2) * 0.8 + [-1, -1];
X3 = randn(num_samples/num_classes, 2) * 0.8 + [1, 0];

X = [X1; X2; X3];
y_labels = [zeros(size(X1, 1), 1); ones(size(X2, 1), 1); 2 * ones(size(X3, 1), 1)];

% One-Hot Encoding
y = zeros(num_samples, num_classes);
for i = 1:num_classes
    y(y_labels == (i-1), i) = 1;
end

cv = cvpartition(num_samples, 'HoldOut', 0.2);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test = X(test_idx, :);
y_test = y(test_idx, :);
y_test_labels = y_labels(test_idx);

lr = 0.05;
epochs = 500;
batch_size = 32;
k_folds = 5;

structure = [size(X_train, 2), 10, size(y_train, 2)];
activations = {'relu', 'softmax'}; 

fprintf('--- run FNN for classification ---\n');
fprintf('Network structure: %s\n', mat2str(structure));

tic;
[params, train_losses, val_losses] = fnn(X_train, y_train, lr, epochs, batch_size, structure, activations, k_folds);
toc;
fprintf('Training finished\n');

figure('Name', 'Loss Curve');
plot(1:epochs, train_losses, 'b', 'LineWidth', 2);
hold on;
plot(1:epochs, val_losses, 'r', 'LineWidth', 2);
title('Loss Curve');
xlabel('Epochs');
ylabel('Loss');
legend('Training loss', 'Validation loss');
grid on;

if structure(1) == 2
    x1_min = min(X(:,1)) - 1; x1_max = max(X(:,1)) + 1;
    x2_min = min(X(:,2)) - 1; x2_max = max(X(:,2)) + 1;
    [x1_grid, x2_grid] = meshgrid(linspace(x1_min, x1_max, 100), linspace(x2_min, x2_max, 100));
    
    X_grid = [x1_grid(:), x2_grid(:)];
    
    [~, A_pred] = forward(X_grid', params, activations, length(structure) - 1);
    
    [~, y_pred_labels] = max(A_pred{end}, [], 1);
    y_pred_labels = reshape(y_pred_labels, size(x1_grid));
    
    figure('Name', 'Decisiong Boundary');

    custom_cmap = [
        [255, 178, 178];    % 红
        [178, 255, 178];    % 绿
        [178, 178, 255]     % 蓝
    ] / 255; 
    
    % 绘制决策边界背景
    contourf(x1_grid, x2_grid, y_pred_labels, num_classes-1, 'LineStyle', 'none');
    colormap(custom_cmap);
    hold on;
    
    % 绘制原始测试数据点
    test_colors = {'r', 'g', 'b'};
    
    for c = 0:num_classes-1
        idx = y_test_labels == c;
        scatter(X_test(idx, 1), X_test(idx, 2), 80, test_colors{c+1}, 'o', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.0, 'DisplayName', ['Class ', num2str(c)]);
    end
    
    title('Decision Boundary');
    xlabel('Feature X1');
    ylabel('Feature X2');
    legend('Location', 'Best');
    axis tight;
    grid on;
end