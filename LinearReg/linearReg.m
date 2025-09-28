function [theta, loss] = linearReg(X, y, lr, epochs, method, batch_size)
%   输入参数:
%   X           : 特征矩阵 (m x n)。m 是样本数，n 是特征数。
%   y           : 目标变量向量 (m x 1)。
%   lr          : 学习率。
%   epochs      : 迭代次数。
%   method      : 梯度下降方法选择 ('BGD', 'SGD', 'MBGD')。
%   batch_size  : 小批量大小（仅用于 'MBGD'，'SGD' 可看作 batch_size=1）。
    
    m = length(y); % 训练样本数
    X = [ones(m, 1), X];
    n = size(X, 2); % 特征数 (包含偏置项)

    % 初始化参数 theta 和损失历史记录
    theta = zeros(n, 1);
    loss = zeros(epochs, 1);

    if strcmp(method, 'MBGD') && (~exist('batch_size', 'var') || isempty(batch_size))
        error('MBGD requires a specified batch_size.');
    elseif strcmp(method, 'SGD')
        batch_size = 1;
    elseif strcmp(method, 'BGD')
        batch_size = m;
    end

    fprintf('开始使用 %s 进行梯度下降...\n', upper(method));

    for iter = 1:epochs
        if strcmp(method, 'BGD')
            sample_indices = 1:m; 
        elseif strcmp(method, 'SGD')
            sample_indices = randi(m, 1); 
        elseif strcmp(method, 'MBGD')
            start_index = randi(m - batch_size + 1, 1);
            sample_indices = start_index : (start_index + batch_size - 1);
        else
            error('未知的方法选择: 请使用 ''BGD'', ''SGD'', 或 ''MBGD''.');
        end

        X_batch = X(sample_indices, :);
        y_batch = y(sample_indices);
        m_batch = length(y_batch);

        h = X_batch * theta;
        errors = h - y_batch; 

        % MSE
        gradient = (1/m_batch) * (X_batch' * errors);
        theta = theta - lr * gradient;
        loss(iter) = computeCost(X, y, theta);

        if mod(iter, round(epochs/10)) == 0 || iter == 1
            fprintf('迭代 %d/%d | 损失 J: %f\n', iter, epochs, loss(iter));
        end

    end

    fprintf('梯度下降完成。\n');
end

% MSE
function J = computeCost(X, y, theta)
    m = length(y); % 样本数

    predictions = X * theta; % m x 1
    sqrErrors = (predictions - y).^2; % m x 1
    J = (1/(2*m)) * sum(sqrErrors);
end