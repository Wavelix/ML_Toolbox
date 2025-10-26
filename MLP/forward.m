function [cache, A] = forward(X, params, activations, num_layers)
    cache = struct();
    A = cell(1, num_layers + 1);
    A{1} = X; 
    
    for l = 1:num_layers
        W_name = ['W' num2str(l)];
        b_name = ['b' num2str(l)];
        
        Z = params.(W_name) * A{l} + params.(b_name);
        [A{l+1}, Z_cache] = activation(Z, activations{l});
        
        % 缓存 Z, A_prev 用于反向传播
        cache.(['Z' num2str(l)]) = Z_cache;
        cache.(['A' num2str(l)]) = A{l};
    end
end