clear; close all; clc;

num_samples = 400;
X = linspace(-5, 5, num_samples)';
y = sin(2*X) + 0.2 * randn(num_samples, 1);

cv = cvpartition(num_samples, 'HoldOut', 0.2);
train_idx = training(cv);
test_idx = test(cv);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test = X(test_idx, :);
y_test = y(test_idx, :);

lr = 0.005;
epochs = 2000;
batch_size = 16;
k_folds = 5;

structure = [size(X_train, 2), 40, 40, size(y_train, 2)];
activations = {'relu', 'relu', 'none'}; 

fprintf('--- run FNN for data fitting ---\n');
fprintf('Network structure: %s\n', mat2str(structure));

tic;
[params, train_losses, val_losses] = fnn(X_train, y_train, lr, epochs, batch_size, structure, activations, k_folds);
toc;
fprintf('Training finishid\n');

figure('Name', 'Loss Curve');
plot(1:epochs, train_losses, 'b', 'LineWidth', 2);
hold on;
plot(1:epochs, val_losses, 'r', 'LineWidth', 2);
title('Loss Curve');
xlabel('Epochs');
ylabel('MSE');
legend('Training loss', 'Validation loss');
grid on;

if structure(1) == 1
    X_full_range = linspace(min(X)-1, max(X)+1, 300)';
    
    [~, A_pred] = forward(X_full_range', params, activations, length(structure) - 1);
    y_pred = A_pred{end}';
    
    figure('Name', 'Data Fitting');
    
    scatter(X_train, y_train, 40, 'b', 'o', 'filled', 'DisplayName', 'Training data');
    hold on;
    scatter(X_test, y_test, 40, 'r', 'x', 'LineWidth', 2, 'DisplayName', 'Test data');
    
    plot(X_full_range, y_pred, 'k-', 'LineWidth', 3, 'DisplayName', 'FNN 拟合曲线');
    
    title('Data Fitting Result');
    xlabel('Feature X');
    ylabel('Target Y');
    legend('Location', 'Best');
    grid on;
end