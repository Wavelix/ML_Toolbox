function [accuracy, y_pred] = knn(X_train, y_train, X_test, y_test, k)
%   输入参数:
%       X_train: 训练集特征矩阵
%       y_train: 训练集标签向量
%       X_test:  测试集特征矩阵
%       y_test:  测试集标签向量
%       k:       近邻数 (k)
%
%   输出参数:
%       accuracy: 模型在测试集上的分类准确率 (0 到 1 之间)
%       y_pred:   预测标签向量

    if size(X_train, 2) ~= size(X_test, 2)
        error('训练集和测试集的特征维度必须一致!');
    end
    
    num_test = size(X_test, 1);
    
    fprintf('使用 KD-Tree 进行 %d 个测试样本的 %d-NN 搜索...\n', num_test, k);
    
    [~, ~, nearest_labels_matrix] = kdtree_search(X_train, y_train, k, X_test);
    
    y_pred = zeros(num_test, 1);
    unique_labels = unique(y_train);

    for i = 1:num_test
        k_neighbors_labels = nearest_labels_matrix(i, :);
        
        vote_counts = zeros(size(unique_labels));
        for j = 1:length(unique_labels)
            label = unique_labels(j);
            vote_counts(j) = sum(k_neighbors_labels == label);
        end
        
        [~, max_idx] = max(vote_counts);
        y_pred(i) = unique_labels(max_idx);
        
    end

    correct_predictions = sum(y_pred == y_test);
    total_samples = length(y_test);
    
    accuracy = correct_predictions / total_samples;
end