function [centroids, labels] = k_means(X, k, max_iters)

%   Input:
%       X - Data matrix, where each row is a data point.
%       k - Number of clusters.
%       max_iters - Maximum number of iterations.
%
%   Output:
%       centroids - A k-by-d matrix, where each row is a centroid of a cluster.
%       labels - A column vector of cluster assignments for each data point.

[m, n] = size(X);
rand_indices = randperm(m);
centroids = X(rand_indices(1:k), :);
labels = zeros(m, 1);

for i = 1:max_iters
    for j = 1:m
        distances = sum((centroids - X(j, :)).^2, 2);
        [~, labels(j)] = min(distances);
    end
    
    new_centroids = zeros(k, n);
    for c = 1:k
        cluster_points = X(labels == c, :);
        if ~isempty(cluster_points)
            new_centroids(c, :) = mean(cluster_points, 1);
        else
            rand_indices = randperm(m);
            new_centroids(c, :) = X(rand_indices(1), :);
        end
    end
    
    if isequal(centroids, new_centroids)
        break;
    end
    
    centroids = new_centroids;
end

end
