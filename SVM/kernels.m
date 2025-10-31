function K = kernels(X1, X2, type, param)
%   returns the kernel matrix of size
%   size(X1,1) x size(X2,1). Supported types:
%     - 'linear': K(x,z) = x*z'
%     - 'gaussian' or 'rbf': K(x,z) = exp(-||x-z||^2 / (2*sigma^2))
%
%     - for 'gaussian', requires field 'sigma' (>0)

if nargin < 3 || isempty(type)
    type = 'linear';
end
if nargin < 4
    param = struct();
end

[~, d1] = size(X1);
[~, d2] = size(X2);
if d1 ~= d2
    error('X1 and X2 must have the same number of columns (features).');
end

switch lower(type)
    case 'linear'
        K = X1 * X2';
    case {'gaussian','rbf'}
        if ~isfield(param, 'sigma') || isempty(param.sigma) || param.sigma <= 0
            error('For gaussian kernel, param.sigma > 0 is required.');
        end
        sigma = param.sigma;
        % Compute squared distances in a vectorized way
        X1_sq = sum(X1.^2, 2);
        X2_sq = sum(X2.^2, 2)';
        % Use broadcasting with bsxfun-like behavior
        dists = (X1_sq + X2_sq) - 2*(X1 * X2');
        % Numerical stability: ensure non-negative
        dists(dists < 0) = 0;
        K = exp(-dists ./ (2*sigma^2));
    otherwise
        error('Unknown kernel type: %s', type);
end
