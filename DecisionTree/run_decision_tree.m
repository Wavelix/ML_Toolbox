clear; close all; clc;

filePath = fullfile('..', 'iris.data');
if ~exist(filePath, 'file')
    filePath2 = 'iris.data';
    if exist(filePath2,'file')
        filePath = filePath2;
    else
        error('未找到 iris.data');
    end
end

fid = fopen(filePath, 'r');
if fid == -1
    error('无法打开数据文件: %s', filePath);
end
C = textscan(fid, '%f%f%f%f%s', 'Delimiter', ',', 'CollectOutput', false, 'ReturnOnError', false);
fclose(fid);

X = [C{1}, C{2}, C{3}, C{4}];
labels = C{5}; % cell array of strings

% 移除可能的空行
valid = all(~isnan(X), 2) & ~cellfun(@isempty, labels);
X = X(valid, :);
labels = labels(valid);

% 将字符串标签映射到整数 1..K
[uniqueLabels, ~, y] = unique(labels);
y = double(y(:));

rng(42);
N = size(X,1);
perm = randperm(N);
X = X(perm, :);
y = y(perm, :);

train_ratio = 0.8;
Ntrain = floor(N * train_ratio);
X_train = X(1:Ntrain, :);
y_train = y(1:Ntrain);
X_test  = X(Ntrain+1:end, :);
y_test  = y(Ntrain+1:end);

params = struct('maxDepth', 10, 'minSamplesSplit', 2, 'minGain', 1e-6);
fprintf('--- Run Decision Tree ---\n');
fprintf('训练集大小: %d | 测试集大小: %d | 特征数: %d\n', size(X_train,1), size(X_test,1), size(X_train,2));

tic;
tree = decision_tree(X_train, y_train, 'maxDepth', params.maxDepth, ...
                                   'minSamplesSplit', params.minSamplesSplit, ...
                                   'minGain', params.minGain);

y_pred = decision_tree(tree, X_test);

elapsed = toc;

accuracy = mean(y_pred == y_test);

fprintf('训练耗时: %.4f s\n', elapsed);
fprintf('Accuracy: %.4f\n', accuracy);
