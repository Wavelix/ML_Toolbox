function grads = backward(AL, Y, cache, params, activations, num_layers, m_batch)
    grads = struct();
    
    % 计算输出层梯度 dZ_L
    if strcmpi(activations{num_layers}, 'sigmoid')
        dZ = AL - Y; 
        
    elseif strcmpi(activations{num_layers}, 'softmax')
        dZ = AL - Y; 
        
    elseif strcmpi(activations{num_layers}, 'none')
        dZ = AL - Y; 
        
    else
        dA = - (Y ./ AL) + ((1-Y) ./ (1-AL)); % Cross-Entropy Loss 对 A_L 的导数
        Z_L = cache.(['Z' num2str(num_layers)]);
        dZ = activation_prime(dA, Z_L, activations{num_layers});
    end
    
    % 反向迭代
    for l = num_layers:-1:1
        A_prev = cache.(['A' num2str(l)]); % A{l}
        W = params.(['W' num2str(l)]);
        
        dW = (1/m_batch) * (dZ * A_prev');
        db = (1/m_batch) * sum(dZ, 2);
        
        % 存储梯度
        grads.(['dW' num2str(l)]) = dW;
        grads.(['db' num2str(l)]) = db;
        
        if l > 1
            % 计算 dZ_prev: dZ^{l-1} = W^{lT} * dZ^l .* g'({Z^{l-1}})
            dA_prev = W' * dZ;
            Z_prev = cache.(['Z' num2str(l-1)]);
            dZ = activation_prime(dA_prev, Z_prev, activations{l-1});
        end
    end
end