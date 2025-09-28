function dZ = activation_prime(dA, Z, type)
    switch lower(type)
        case 'relu'
            dZ = dA .* (Z > 0);
        case 'tanh'
            A = tanh(Z);
            dZ = dA .* (1 - A.^2);
        case 'sigmoid'
            A = 1 ./ (1 + exp(-Z));
            dZ = dA .* A .* (1 - A);
        case {'softmax', 'none'} 
            % Softmax 的导数合并在后向传播的 dZ_L 计算中
            dZ = dA; 
        otherwise
            error('不支持的激活函数类型: %s', type);
    end
end