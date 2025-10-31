function model = svm_train(X, y, C, kernelType, kernelParam, tol, max_passes)
%   Train an SVM classifier using SMO
%   - X: n x d input matrix
%   - y: n x 1 labels in {-1, +1}
%   - C: regularization parameter (>0)
%   - kernelType: 'linear' or 'gaussian' (aka 'rbf')
%   - kernelParam: struct with fields depending on kernel (e.g., sigma for gaussian)
%   - tol: tolerance for KKT conditions (default 1e-3)
%   - max_passes: max number of passes without alpha changes (default 5)
%
%   Returns a struct model containing:
%     .alphas (n x 1), .b (scalar), .kernelType, .kernelParam
%     .sv_X, .sv_y, .sv_alphas (support vectors and corresponding alphas)
%     .w (d x 1) if linear kernel

if nargin < 3 || isempty(C), C = 1; end
if nargin < 4 || isempty(kernelType), kernelType = 'linear'; end
if nargin < 5 || isempty(kernelParam), kernelParam = struct(); end
if nargin < 6 || isempty(tol), tol = 1e-3; end
if nargin < 7 || isempty(max_passes), max_passes = 5; end

X = double(X);
y = double(y(:));
[n, ~] = size(X);

uy = unique(y);
if numel(uy) ~= 2 || ~all(ismember(uy, [-1, 1]))
    error('Labels y must be in {-1, +1}.');
end

alphas = zeros(n,1);
b = 0;

K = kernels(X, X, kernelType, kernelParam);

passes = 0;
% num_changed_alphas = 0;

while passes < max_passes
    num_changed_alphas = 0;
    for i = 1:n
        f_i = sum(alphas .* y .* K(:, i)) + b;
        E_i = f_i - y(i);

        if ( (y(i)*E_i < -tol && alphas(i) < C) || (y(i)*E_i > tol && alphas(i) > 0) )
            % Select j != i randomly
            j = randi(n);
            while j == i
                j = randi(n);
            end

            f_j = sum(alphas .* y .* K(:, j)) + b;
            E_j = f_j - y(j);

            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);

            % Compute L and H
            if y(i) ~= y(j)
                L = max(0, alpha_j_old - alpha_i_old);
                H = min(C, C + alpha_j_old - alpha_i_old);
            else
                L = max(0, alpha_i_old + alpha_j_old - C);
                H = min(C, alpha_i_old + alpha_j_old);
            end
            if L == H
                continue;
            end

            % Compute eta
            eta = 2*K(i,j) - K(i,i) - K(j,j);
            if eta >= 0
                continue;
            end

            % Update alpha_j
            alphas(j) = alpha_j_old - y(j)*(E_i - E_j)/eta;

            % Clip
            if alphas(j) > H
                alphas(j) = H;
            elseif alphas(j) < L
                alphas(j) = L;
            end

            if abs(alphas(j) - alpha_j_old) < 1e-5
                alphas(j) = alpha_j_old;
                continue;
            end

            % Update alpha_i
            alphas(i) = alpha_i_old + y(i)*y(j)*(alpha_j_old - alphas(j));

            % Compute b1 and b2
            b1 = b - E_i - y(i)*(alphas(i)-alpha_i_old)*K(i,i) - y(j)*(alphas(j)-alpha_j_old)*K(i,j);
            b2 = b - E_j - y(i)*(alphas(i)-alpha_i_old)*K(i,j) - y(j)*(alphas(j)-alpha_j_old)*K(j,j);

            % Update b
            if (alphas(i) > 0) && (alphas(i) < C)
                b = b1;
            elseif (alphas(j) > 0) && (alphas(j) < C)
                b = b2;
            else
                b = 0.5*(b1 + b2);
            end

            num_changed_alphas = num_changed_alphas + 1;
        end
    end

    if num_changed_alphas == 0
        passes = passes + 1;
    else
        passes = 0;
    end
end

% Build model
sv = alphas > 1e-6;
model = struct();
model.alphas = alphas;
model.b = b;
model.kernelType = kernelType;
model.kernelParam = kernelParam;
model.sv_X = X(sv, :);
model.sv_y = y(sv);
model.sv_alphas = alphas(sv);

if strcmpi(kernelType, 'linear')
    % w = sum_i alpha_i y_i x_i
    model.w = (X' * (alphas .* y));
else
    model.w = [];
end

end
