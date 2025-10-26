function [Z, W, mu, evals, explained, X_rec, rec_mse, rec_rmse] = pca(X, k, varargin)

% Inputs:
%   X  - (n x d) data matrix, rows are samples, columns are features
%   k  - target dimension
% options:
%   'Center'       - mean-center the data (default: true)
%   'Standardize'  - z-score features before PCA (default: true)
%
% Outputs:
%   Z         - (n x k) projected data in the k-dimensional PCA subspace
%   W         - (d x k) top-k principal component directions (eigenvectors)
%   mu        - (1 x d) mean of original X used for centering
%   evals     - (d x 1) eigenvalues of covariance matrix, sorted desc
%   explained - (k x 1) fraction of variance explained by each selected PC
%   X_rec     - (n x d) reconstructed data from Z back to original space
%   rec_mse   - scalar
%   rec_rmse  - scalar

p = inputParser;
p.addParameter('Center', true, @(b)islogical(b) && isscalar(b));
p.addParameter('Standardize', true, @(b)islogical(b) && isscalar(b));
p.parse(varargin{:});
opts = p.Results;

[n, d] = size(X);
if ~(isscalar(k) && isnumeric(k) && k >= 1 && k <= d)
    error('k must be an integer in [1, %d]', d);
end

if opts.Center
    mu = mean(X, 1);
else
    mu = zeros(1, d);
end
Xc = bsxfun(@minus, X, mu);

if opts.Standardize
    sigma = std(Xc, 0, 1);   % 1 x d
    sigma_safe = sigma;
    sigma_safe(sigma_safe == 0) = 1; 
    Xs = bsxfun(@rdivide, Xc, sigma_safe);
else
    sigma = ones(1, d);
    Xs = Xc;
end

C = (Xs' * Xs) / (n - 1);

[V, D] = eig(C);
[evals, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% top-k components
W = V(:, 1:k);

Z = Xs * W;  % n x k

% Reconstruct from k-dim subspace
Xs_rec = Z * W';           % n x d (standardized, centered)
Xc_rec = bsxfun(@times, Xs_rec, sigma); % undo standardization
X_rec  = bsxfun(@plus, Xc_rec, mu);     % undo centering

err = X - X_rec;
rec_mse  = mean(err(:).^2);
rec_rmse = sqrt(rec_mse);

totalVar = sum(evals);
explained = evals(1:k) / totalVar;
end
