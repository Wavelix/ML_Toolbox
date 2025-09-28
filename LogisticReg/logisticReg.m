function [theta, train_losses, val_losses] = logisticReg(X, y, lr, epochs, method, batch_size, k_folds)
%   输入参数：
%       X: 特征矩阵 (m x n)，m为样本数，n为特征数
%       y: 标签向量 (m x 1)，0或1
%       lr: 学习率
%       epochs: 迭代次数
%       method: 梯度下降方法 ('sgd', 'mbgd', 'bgd')
%       batch_size: 小批量梯度下降的批量大小
%       k_folds: K折交叉验证的折数
%   输出参数：
%       theta: 训练得到的模型参数
%       train_losses: 每次迭代的平均训练损失
%       val_losses: 每次迭代的平均验证损失

    X = [ones(size(X, 1), 1), X];
    m = size(X, 1);
    n = size(X, 2);
    
    theta = zeros(n, 1);
    
    train_losses = zeros(epochs, 1);
    val_losses = zeros(epochs, 1);
    
    % K折交叉验证
    cv = cvpartition(m, 'KFold', k_folds);
    
    for epoch = 1:epochs
        epoch_train_loss = 0;
        epoch_val_loss = 0;
        
        for k = 1:k_folds
            train_idx = training(cv, k);
            val_idx = test(cv, k);
            
            X_train = X(train_idx, :);
            y_train = y(train_idx);
            
            X_val = X(val_idx, :);
            y_val = y(val_idx);
            
            m_train = size(X_train, 1);
            
            switch method
                case 'bgd' 
                    h = sigmoid(X_train * theta);
                    grad = (1/m_train) * X_train' * (h - y_train);
                    theta = theta - lr * grad;
                    
                case 'sgd' 
                    idx = randperm(m_train);
                    X_shuffled = X_train(idx, :);
                    y_shuffled = y_train(idx);
                    
                    for i = 1:m_train
                        xi = X_shuffled(i, :);
                        yi = y_shuffled(i);
                        
                        h = sigmoid(xi * theta);
                        grad = xi' * (h - yi);
                        theta = theta - lr * grad;
                    end
                    
                case 'mbgd' 
                    idx = randperm(m_train);
                    X_shuffled = X_train(idx, :);
                    y_shuffled = y_train(idx);
                    
                    for i = 1:batch_size:m_train
                        batch_end = min(i + batch_size - 1, m_train);
                        X_batch = X_shuffled(i:batch_end, :);
                        y_batch = y_shuffled(i:batch_end);
                        
                        m_batch = size(X_batch, 1);
                        h = sigmoid(X_batch * theta);
                        grad = (1/m_batch) * X_batch' * (h - y_batch);
                        theta = theta - lr * grad;
                    end
            end
            
            epoch_train_loss = epoch_train_loss + cross_entropy_loss(X_train, y_train, theta);
            epoch_val_loss = epoch_val_loss + cross_entropy_loss(X_val, y_val, theta);
        end
        
        train_losses(epoch) = epoch_train_loss / k_folds;
        val_losses(epoch) = epoch_val_loss / k_folds;
    end
end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

function J = cross_entropy_loss(X, y, theta)
    m = size(X, 1);
    h = sigmoid(X * theta);
    J = (-1/m) * sum(y .* log(h) + (1-y) .* log(1-h));
end