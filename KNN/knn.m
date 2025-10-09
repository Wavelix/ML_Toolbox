function accuracy = knn(X_train, y_train, X_test, y_test, k)
%   输入参数:
%       X_train: 训练集特征矩阵 (m_train x n)
%       y_train: 训练集标签向量 (m_train x 1)
%       X_test:  测试集特征矩阵 (m_test x n)
%       y_test:  测试集标签向量 (m_test x 1)
%       k:       近邻数 (k)
    
    try
        Mdl = fitcknn(X_train, y_train, 'NumNeighbors', k, ...
                      'Distance', 'euclidean', ...
                      'NSMethod', 'kdtree');
    catch
        error('无法创建 k-NN 模型。请确保已安装 Statistics and Machine Learning Toolbox。');
    end

    y_pred = predict(Mdl, X_test);

    correct_predictions = sum(y_pred == y_test);
    total_samples = length(y_test);
    
    accuracy = correct_predictions / total_samples;
    
    fprintf('k = %d 时，测试集准确率: %.4f\n', k, accuracy);
end