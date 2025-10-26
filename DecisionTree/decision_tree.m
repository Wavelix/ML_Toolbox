function out = decision_tree(A, B, varargin)

%   训练：
%       tree = decision_tree(X, y, 'Name', Value, ...)
%   预测：
%       y_pred = decision_tree(tree, X)
%
%   可选训练参数（以'Name',Value形式传入）：
%     'maxDepth'        (默认 10)         最大树深度
%     'minSamplesSplit' (默认 2)          继续分裂所需的最小样本数
%     'minGain'         (默认 1e-6)       最小信息增益阈值（小于此值不再分裂）
%
%   训练返回 tree（结构体）：
%     isLeaf: 是否为叶子
%     class:  叶子节点的类别（从1开始的整数）
%     featureIndex: 用于划分的特征索引
%     threshold:    连续特征的二分阈值 x(feature) <= threshold 走左子树，否则右子树
%     left, right:  子树（结构体）

    % 若第一个参数为树结构体，则执行预测模式，否则执行训练模式
    if isstruct(A) && isfield(A, 'isLeaf')
        % 预测模式
        tree = A;
        X = B;
        out = decision_tree_predict(tree, X);
        return;
    end

    % 训练模式
    X = A;
    y = B;

    p = inputParser;
    addParameter(p, 'maxDepth', 10);
    addParameter(p, 'minSamplesSplit', 2);
    addParameter(p, 'minGain', 1e-6);
    parse(p, varargin{:});
    params = p.Results;

    % 确保 y 为列向量的正整数标签（1..K）
    if isrow(y), y = y(:); end
    if ~isnumeric(y)
        error('y 必须为数值型标签（1..K）。');
    end

    out = build_tree(X, y, 0, params);
end

function node = build_tree(X, y, depth, params)
    numSamples = size(X, 1);
    classes = unique(y);

    % 如果全为同一类，直接叶子
    if numel(classes) == 1
        node = make_leaf(classes(1));
        return;
    end

    % 达到最大深度
    if depth >= params.maxDepth
        node = make_leaf(majority_class(y));
        return;
    end

    % 样本数量过少
    if numSamples < params.minSamplesSplit
        node = make_leaf(majority_class(y));
        return;
    end

    % 搜索最佳划分
    [bestFeature, bestThreshold, bestGain, leftIdx, rightIdx] = find_best_split(X, y);

    % 信息增益不足
    if isempty(bestFeature) || bestGain < params.minGain
        node = make_leaf(majority_class(y));
        return;
    end

    % 递归构建左右子树
    leftChild = build_tree(X(leftIdx, :), y(leftIdx), depth + 1, params);
    rightChild = build_tree(X(rightIdx, :), y(rightIdx), depth + 1, params);

    node = struct();
    node.isLeaf = false;
    node.class = [];
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
    node.left = leftChild;
    node.right = rightChild;
end

function [bestFeature, bestThreshold, bestGain, bestLeftIdx, bestRightIdx] = find_best_split(X, y)
    % 穷举每个特征的候选阈值，计算信息增益，选择最佳
    [n, d] = size(X);
    baseH = entropy_y(y);

    bestGain = -inf;
    bestFeature = [];
    bestThreshold = [];
    bestLeftIdx = [];
    bestRightIdx = [];

    for j = 1:d
        xj = X(:, j);
        % 将 NaN 放到一侧
        nanMask = isnan(xj);
        if all(nanMask)
            continue; % 全是 NaN 无法分
        end
        xj_valid = xj(~nanMask);
        y_valid = y(~nanMask);

        % 唯一值过少无法产生阈值
        ux = unique(xj_valid);
        if numel(ux) < 2
            continue;
        end
        % 候选阈值：相邻唯一值的中点
        thr = (ux(1:end-1) + ux(2:end)) / 2;

        for t = 1:numel(thr)
            th = thr(t);
            leftIdx_valid = xj_valid <= th;
            rightIdx_valid = ~leftIdx_valid;
            nL = sum(leftIdx_valid);
            nR = sum(rightIdx_valid);
            if nL == 0 || nR == 0
                continue;
            end

            yL = y_valid(leftIdx_valid);
            yR = y_valid(rightIdx_valid);

            HL = entropy_y(yL);
            HR = entropy_y(yR);

            % 加权后验熵
            postH = (nL / numel(y_valid)) * HL + (nR / numel(y_valid)) * HR;
            gain = baseH - postH;

            if gain > bestGain
                % 在原始索引上形成对应划分
                bestGain = gain;
                bestFeature = j;
                bestThreshold = th;

                leftAll = false(n,1);
                rightAll = false(n,1);
                % NaN 的样本统一放到右侧
                leftAll(~nanMask) = leftIdx_valid;
                rightAll(~nanMask) = rightIdx_valid;
                rightAll(nanMask) = true;

                bestLeftIdx = leftAll;
                bestRightIdx = rightAll;
            end
        end
    end

    if isinf(bestGain)
        bestFeature = [];
        bestThreshold = [];
        bestLeftIdx = [];
        bestRightIdx = [];
        bestGain = -inf;
    end
end

function H = entropy_y(y)
    % 计算标签 y 的信息熵（以2为底）
    if isempty(y)
        H = 0;
        return;
    end
    % 统计类别频次
    u = unique(y);
    n = numel(y);
    counts = zeros(numel(u),1);
    for i = 1:numel(u)
        counts(i) = sum(y == u(i));
    end
    p = counts / n;
    % 避免 log2(0)
    H = -sum(p .* log2(p + eps));
end

function c = majority_class(y)
    % 返回出现频率最高的类别（若并列，取最小标签值）
    u = unique(y);
    counts = zeros(numel(u),1);
    for i = 1:numel(u)
        counts(i) = sum(y == u(i));
    end
    [~, idx] = max(counts);
    c = u(idx);
end

function node = make_leaf(c)
    node = struct('isLeaf', true, 'class', c, 'featureIndex', [], 'threshold', [], 'left', [], 'right', []);
end

function y_pred = decision_tree_predict(tree, X)
%DECISION_TREE_PREDICT 使用训练好的树对样本进行预测
%   y_pred = decision_tree_predict(tree, X)
    n = size(X,1);
    y_pred = zeros(n,1);
    for i = 1:n
        y_pred(i) = predict_one(tree, X(i,:));
    end
end

function c = predict_one(node, x)
    while ~node.isLeaf
        f = node.featureIndex;
        t = node.threshold;
        v = x(f);
        if isnan(v)
            % NaN 走右子树
            node = node.right;
        elseif v <= t
            node = node.left;
        else
            node = node.right;
        end
    end
    c = node.class;
end
