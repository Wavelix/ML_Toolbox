function [nodes, root_idx, nearest_labels] = kdtree_search(X, y, k, X_query)

%   输入:
%       X: 训练集特征矩阵 (m x n)
%       y: 训练集标签向量 (m x 1)
%       k: 要查找的近邻数量
%       X_query: 查询点矩阵 (m_query x n)
%   输出:
%       nodes: 构建的 KD-Tree 节点结构体数组
%       root_idx: 根节点索引
%       nearest_labels: 每个查询点找到的 k 个近邻的标签 (m_query x k)

    k_dim = size(X, 2);
    num_query = size(X_query, 1);
    
    nodes_cell = {};
    node_count = 0;
    
    function node = create_node()
        node = struct('point', [], 'label', [], 'index', [], ...
                      'depth', 0, 'split_dim', 0, ...
                      'left_child', 0, 'right_child', 0);
    end

    function current_idx = build_recursive(data_indices, depth)
        
        if isempty(data_indices)
            current_idx = 0;
            return;
        end
        
        split_dim = mod(depth, k_dim) + 1;
        current_X = X(data_indices, :);
        [~, sorted_idx] = sort(current_X(:, split_dim));
        median_index_in_sorted = floor(length(data_indices) / 2) + 1;
        median_data_idx = data_indices(sorted_idx(median_index_in_sorted));
        
        node_count = node_count + 1;
        current_idx = node_count;
        
        new_node = create_node();
        new_node.point = X(median_data_idx, :)';
        new_node.label = y(median_data_idx);
        new_node.index = median_data_idx;
        new_node.depth = depth;
        new_node.split_dim = split_dim;
        
        other_indices = data_indices(sorted_idx([1:median_index_in_sorted-1, median_index_in_sorted+1:end]));
        
        left_indices = [];
        right_indices = [];
        median_val = X(median_data_idx, split_dim);
        
        for idx = other_indices'
            if X(idx, split_dim) < median_val
                left_indices = [left_indices, idx];
            else 
                right_indices = [right_indices, idx];
            end
        end

        new_node.left_child = build_recursive(left_indices, depth + 1);
        new_node.right_child = build_recursive(right_indices, depth + 1);
        
        nodes_cell{current_idx} = new_node;
    end
    
    all_indices = 1:size(X, 1);
    root_idx = build_recursive(all_indices, 0);
    nodes = [nodes_cell{:}];
    
    nearest_labels = zeros(num_query, k);
    
    % 欧氏距离
    function dist = euclidean_distance(p1, p2)
        dist = sqrt(sum((p1 - p2).^2));
    end
    
    % 插入结果到优先队列
    function results = insert_result(results, dist, index, label, k_val)
        if length(results) >= k_val && dist >= results{end, 1}
            return;
        end
        
        new_entry = {dist, index, label};
        results = [results; new_entry];
        
        [~, sort_idx] = sort([results{:, 1}]);
        results = results(sort_idx, :);
        
        if length(results) > k_val
            results = results(1:k_val, :);
        end
    end

    % 递归搜索
    function results = search_recursive(node_idx, results, q_point_vec)
        
        if node_idx == 0
            return;
        end
        
        node = nodes(node_idx);
        
        % 1. 计算当前点距离并更新结果列表
        current_dist = euclidean_distance(q_point_vec, node.point);
        results = insert_result(results, current_dist, node.index, node.label, k);
        
        if isempty(results)
            max_dist = inf;
        else
            max_dist = results{end, 1};
        end
        
        % 2. 确定搜索顺序
        split_dim = node.split_dim;
        
        if q_point_vec(split_dim) < node.point(split_dim)
            search_first_idx = node.left_child;
            search_second_idx = node.right_child;
        else
            search_first_idx = node.right_child;
            search_second_idx = node.left_child;
        end
        
        % 3. 递归搜索主要分支
        results = search_recursive(search_first_idx, results, q_point_vec);
        
        % 4. 检查是否需要回溯
        plane_dist = abs(q_point_vec(split_dim) - node.point(split_dim));
        
        if ~isempty(results)
             max_dist = results{end, 1};
        else
             max_dist = inf;
        end

        if plane_dist < max_dist
            results = search_recursive(search_second_idx, results, q_point_vec);
        end
    end
    
    % 对每一个查询点进行搜索
    for i = 1:num_query
        q_point = X_query(i, :)';
        results = cell(0, 3); 
        
        results = search_recursive(root_idx, results, q_point);
        
        if size(results, 1) < k
            % 如果找到的近邻数少于 k，用 0 填充剩余部分
            labels_found = cell2mat(results(:, 3));
            nearest_labels(i, 1:size(labels_found, 1)) = labels_found;
        else
            nearest_labels(i, :) = cell2mat(results(:, 3));
        end
    end
end