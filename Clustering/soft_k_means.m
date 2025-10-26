function [centroids, responsibilities] = soft_k_means(X, k, max_iters, beta)

%   Input:
%       X - Data matrix, where each row is a data point.
%       k - Number of clusters.
%       max_iters - Maximum number of iterations.
%       beta - Stiffness parameter. Higher beta makes the assignment harder.
%
%   Output:
%       centroids - A k-by-d matrix, where each row is a centroid of a cluster.
%       responsibilities - An m-by-k matrix of responsibilities.

[m, n] = size(X);
rand_indices = randperm(m);
centroids = X(rand_indices(1:k), :);
responsibilities = zeros(m, k);

for i = 1:max_iters
    for j = 1:m
        distances = sum((centroids - X(j, :)).^2, 2);
        exp_neg_beta_dist = exp(-beta * distances);
        responsibilities(j, :) = exp_neg_beta_dist / sum(exp_neg_beta_dist);
    end
    
    new_centroids = zeros(k, n);
    for c = 1:k
        numerator = sum(responsibilities(:, c) .* X, 1);
        denominator = sum(responsibilities(:, c));
        new_centroids(c, :) = numerator / denominator;
    end
    
    if norm(centroids - new_centroids, 'fro') < 1e-4
        break;
    end
    
    centroids = new_centroids;
end

end
