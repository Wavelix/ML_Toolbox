function [A, Z] = activation(Z, type)
    Z = Z; % 缓存 Z
    switch lower(type)
        case 'relu'
            A = max(0, Z);
        case 'tanh'
            A = tanh(Z);
        case 'sigmoid'
            A = 1 ./ (1 + exp(-Z));
        case 'softmax'
            Z_shift = Z - max(Z);
            exp_Z = exp(Z_shift);
            A = exp_Z ./ sum(exp_Z, 1);
        case 'none' % 线性输出
            A = Z;
        otherwise
            error('不支持的激活函数类型: %s', type);
    end
end