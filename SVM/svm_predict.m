function [pred, f] = svm_predict(model, X)
%     - pred: predicted labels in {-1, +1}
%     - f: decision function values

X = double(X);
% [n, ~] = size(X);

if isfield(model, 'w') && ~isempty(model.w) && strcmpi(model.kernelType, 'linear')
    f = X * model.w + model.b;
else
    Ktest = kernels(X, model.sv_X, model.kernelType, model.kernelParam);
    f = Ktest * (model.sv_alphas .* model.sv_y) + model.b;
end

pred = sign(f);
pred(pred==0) = 1;

end
