function J = compute_loss(AL, Y, output_activation)
    m = size(Y, 2);
    
    if strcmpi(output_activation, 'sigmoid') % 二分类
        AL = max(min(AL, 1 - 1e-10), 1e-10); 
        J = (-1/m) * sum(sum(Y .* log(AL) + (1-Y) .* log(1-AL)));
        
    elseif strcmpi(output_activation, 'softmax') % 多分类
        AL = max(AL, 1e-10);
        J = (-1/m) * sum(sum(Y .* log(AL)));
        
    elseif strcmpi(output_activation, 'none') % 回归
        J = (1/(2*m)) * sum(sum((AL - Y).^2));
        
    else
        error('未知的输出层激活函数或损失类型。');
    end
end