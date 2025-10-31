function plot_decision_views(X, y, model, text)

if size(X,2) ~= 2
    error('plot_decision_views supports only 2D data.');
end

pad = 0.5;
x1_min = min(X(:,1)) - pad; x1_max = max(X(:,1)) + pad;
x2_min = min(X(:,2)) - pad; x2_max = max(X(:,2)) + pad;

step = 200;
x1 = linspace(x1_min, x1_max, step);
x2 = linspace(x2_min, x2_max, step);
[Xg1, Xg2] = meshgrid(x1, x2);
Xgrid = [Xg1(:), Xg2(:)];

[~, fvals] = svm_predict(model, Xgrid);
F = reshape(fvals, size(Xg1));

% support vectors
sv_mask = false(size(y));
if isfield(model,'sv_X')
    tol = 1e-10;
    for i = 1:size(model.sv_X,1)
        diffs = sum(abs(X - model.sv_X(i,:)) < tol, 2) == 2;
        sv_mask = sv_mask | diffs;
    end
end

clf;
colormap();

% Decision function view
% subplot(1,2,1);
contourf(Xg1, Xg2, F, 30, 'LineColor', 'none');
hold on;
contour(Xg1, Xg2, F, [0 0], 'k-', 'LineWidth', 2);
scatter(X(y==1,1), X(y==1,2), 36, 'r', 'filled');
scatter(X(y==-1,1), X(y==-1,2), 36, 'b', 'filled');
% if any(sv_mask)
%     scatter(X(sv_mask,1), X(sv_mask,2), 80, 'yo', 'LineWidth', 1.5);
% end
hold off;
axis tight; axis equal;
title(sprintf('%s - Decision Function view',text));
% legend({'f(x)','f(x)=0','y=+1','y=-1','SV'});

% % 0-1 view
% subplot(1,2,2);
% Z = sign(F);
% Z(Z==0) = 1;
% imagesc(x1, x2, Z);
% set(gca,'YDir','normal');
% hold on;
% contour(Xg1, Xg2, F, [0 0], 'k-', 'LineWidth', 2);
% scatter(X(y==1,1), X(y==1,2), 36, 'r', 'filled');
% scatter(X(y==-1,1), X(y==-1,2), 36, 'b', 'filled');
% % if any(sv_mask)
% %     scatter(X(sv_mask,1), X(sv_mask,2), 80, 'yo', 'LineWidth', 1.5);
% % end
% hold off;
% axis tight; axis equal;
% title(sprintf('%s - 0-1 View',text));
% % legend({'f(x)=0','y=+1','y=-1','SV'});

end
