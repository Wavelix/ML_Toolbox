function [centroids, labels] = k_means_pp(X, k, max_iters)

%   Input:
%       X - Data matrix, where each row is a data point.
%       k - Number of clusters.
%       max_iters - Maximum number of iterations for the K-Means part.
%
%   Output:
%       centroids - A k-by-d matrix, where each row is a centroid of a cluster.
%       labels - A column vector of cluster assignments for each data point.

[m, n] = size(X);
centroids = zeros(k, n);

centroids(1, :) = X(randi(m), :);

distances = inf(m, 1);
for i = 2:k
    dist_to_last_centroid = sum((X - centroids(i-1, :)).^2, 2);
    distances = min(distances, dist_to_last_centroid);
    
    prob = distances / sum(distances);
    cumulative_prob = cumsum(prob);
    r = rand();
    new_centroid_idx = find(r < cumulative_prob, 1, 'first');
    centroids(i, :) = X(new_centroid_idx, :);
end

labels = zeros(m, 1);
for i = 1:max_iters
    for j = 1:m
        dist = sum((centroids - X(j, :)).^2, 2);
        [~, labels(j)] = min(dist);
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
