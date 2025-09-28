function [params, train_losses, val_losses] = fnn(X, y, lr, epochs, batch_size, structure, activations, k_folds)
%   输入参数：
%       X: 特征矩阵 (m x n_input)
%       y: 标签矩阵 (m x n_output)
%       lr: 学习率
%       epochs: 迭代次数
%       batch_size: 小批量梯度下降的批量大小
%       structure: 网络结构 [n_input, n_hidden1, n_hidden2, ..., n_output]
%       activations: 每层的激活函数类型 Cell Array {'relu', 'tanh', 'sigmoid', 'softmax'，'none'}
%       k_folds: K折交叉验证的折数
%   输出参数：
%       params: 训练得到的模型参数结构体
%       train_losses: 每次迭代的平均训练损失
%       val_losses: 每次迭代的平均验证损失
    
    m = size(X, 1);
    num_layers = length(structure) - 1; % 权重矩阵的数量
    
    if length(activations) ~= num_layers
        error('激活函数列表长度必须等于网络层数 (structure 长度减 1)。');
    end

    params = struct();
    for l = 1:num_layers
        % 使用 He 或 Xavier 初始化
        if strcmpi(activations{l}, 'relu')
            % He initialization (适用于 ReLU)
            initializer = sqrt(2 / structure(l)); 
        else
            % Xavier initialization (适用于 tanh/sigmoid)
            initializer = sqrt(1 / structure(l)); 
        end
        
        W = initializer * randn(structure(l+1), structure(l));
        b = zeros(structure(l+1), 1);
        
        params.(['W' num2str(l)]) = W;
        params.(['b' num2str(l)]) = b;
    end
    
    % 初始化损失记录
    train_losses = zeros(epochs, 1);
    val_losses = zeros(epochs, 1);
    
    cv = cvpartition(m, 'KFold', k_folds);
    
    for epoch = 1:epochs
        epoch_train_loss = 0;
        epoch_val_loss = 0;
        
        for k = 1:k_folds
            train_idx = training(cv, k);
            val_idx = test(cv, k);
            
            X_train = X(train_idx, :);
            y_train = y(train_idx, :);
            
            X_val = X(val_idx, :);
            y_val = y(val_idx, :);
            
            m_train = size(X_train, 1);
            
            idx = randperm(m_train);
            X_shuffled = X_train(idx, :);
            y_shuffled = y_train(idx, :);
            
            for i = 1:batch_size:m_train
                batch_end = min(i + batch_size - 1, m_train);
                X_batch = X_shuffled(i:batch_end, :);
                y_batch = y_shuffled(i:batch_end, :);
                
                m_batch = size(X_batch, 1);
                
                [cache, A] = forward(X_batch', params, activations, num_layers);
              
                loss = compute_loss(A{num_layers+1}, y_batch', activations{num_layers});
                
                grads = backward(A{num_layers+1}, y_batch', cache, params, activations, num_layers, m_batch);
                
                for l = 1:num_layers
                    W_name = ['W' num2str(l)];
                    b_name = ['b' num2str(l)];
                    
                    params.(W_name) = params.(W_name) - lr * grads.(['dW' num2str(l)]);
                    params.(b_name) = params.(b_name) - lr * grads.(['db' num2str(l)]);
                end
                
                epoch_train_loss = epoch_train_loss + loss;
            end
            
            [~, A_val] = forward(X_val', params, activations, num_layers);
            val_loss = compute_loss(A_val{num_layers+1}, y_val', activations{num_layers});
            epoch_val_loss = epoch_val_loss + val_loss;
        end
        
        train_losses(epoch) = epoch_train_loss / (k_folds * ceil(m_train / batch_size));
        val_losses(epoch) = epoch_val_loss / k_folds;

        if mod(epoch, round(epochs/10)) == 0 || epoch == 1
            fprintf('迭代 %d/%d | 损失 J: %f\n', epoch, epochs, train_losses(epoch));
        end
    end
end