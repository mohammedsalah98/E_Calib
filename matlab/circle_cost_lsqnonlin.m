%% circle_cost function
function [residuals, jac] = circle_cost_lsqnonlin(events_set, solution_vector)
r = solution_vector(1);
h = solution_vector(2);
k = solution_vector(3);
z = solution_vector(6);
theta = solution_vector(4);
beta = solution_vector(5);
x = events_set(:,1);
y = events_set(:,2);
t = events_set(:,3);
% Find cost
R_sb = rotx(rad2deg(solution_vector(5))) * roty(rad2deg(solution_vector(4)));
t_sb = [solution_vector(2);solution_vector(3);solution_vector(6)];
T_sb = [R_sb t_sb;0 0 0 1];
rotated_pts = T_sb\[events_set';ones(1,height(events_set))];
residuals = rotated_pts(1,:).^2 + rotated_pts(2,:).^2 - solution_vector(1)^2;
outliers_idx = isoutlier(residuals, 'quartiles');
residuals(outliers_idx == 1) = 0;
% Find jacobian
jac = zeros(height(events_set), length(solution_vector));
jac(:,1) = (-2*r) .* ones(height(events_set), 1);
jac(:,2) = (2.*(cos(beta).^2*cos(theta) + sin(beta).^2.*cos(theta)).*(t.*cos(beta).*sin(theta) - z.*cos(beta).*sin(theta) + k.*sin(beta).*sin(theta) - y.*sin(beta).*sin(theta) + h.*cos(beta).^2.*cos(theta) + h.*sin(beta).^2.*cos(theta) - x.*cos(beta).^2.*cos(theta) - x.*sin(beta).^2.*cos(theta)))./((cos(beta).^2 + sin(beta).^2).^2.*(cos(theta).^2 + sin(theta).^2).^2);
jac(:,3) = (2.*cos(beta).*(k.*cos(beta) - y.*cos(beta) - t.*sin(beta) + z.*sin(beta)))./(cos(beta).^2 + sin(beta).^2).^2 + (2.*sin(beta).*sin(theta).*(t.*cos(beta).*sin(theta) - z.*cos(beta).*sin(theta) + k.*sin(beta).*sin(theta) - y.*sin(beta).*sin(theta) + h.*cos(beta).^2.*cos(theta) + h.*sin(beta).^2.*cos(theta) - x.*cos(beta).^2.*cos(theta) - x.*sin(beta).^2.*cos(theta)))./((cos(beta).^2 + sin(beta).^2).^2.*(cos(theta).^2 + sin(theta).^2).^2);
jac(:,4) = (2.*(t.*cos(beta).*cos(theta) - z.*cos(beta).*cos(theta) + k.*sin(beta).*cos(theta) - y.*sin(beta).*cos(theta) - h.*cos(beta)^2.*sin(theta) - h.*sin(beta).^2.*sin(theta) + x.*cos(beta).^2.*sin(theta) + x.*sin(beta).^2.*sin(theta)).*(t.*cos(beta).*sin(theta) - z.*cos(beta).*sin(theta) + k.*sin(beta).*sin(theta) - y.*sin(beta).*sin(theta) + h.*cos(beta).^2*cos(theta) + h.*sin(beta).^2.*cos(theta) - x.*cos(beta).^2.*cos(theta) - x.*sin(beta).^2.*cos(theta)))./((cos(beta).^2 + sin(beta).^2).^2.*(cos(theta).^2 + sin(theta).^2).^2);
jac(:,5) = (2.*(k.*cos(beta).*sin(theta) - y.*cos(beta).*sin(theta) - t.*sin(beta).*sin(theta) + z.*sin(beta).*sin(theta)).*(t.*cos(beta).*sin(theta) - z.*cos(beta).*sin(theta) + k.*sin(beta).*sin(theta) - y.*sin(beta).*sin(theta) + h.*cos(beta).^2.*cos(theta) + h.*sin(beta).^2.*cos(theta) - x.*cos(beta).^2.*cos(theta) - x.*sin(beta).^2.*cos(theta)))./((cos(beta).^2 + sin(beta).^2).^2.*(cos(theta).^2 + sin(theta).^2).^2) - (2.*(k.*cos(beta) - y.*cos(beta) - t.*sin(beta) + z.*sin(beta)).*(t.*cos(beta) - z.*cos(beta) + k.*sin(beta) - y.*sin(beta)))./(cos(beta).^2 + sin(beta).^2).^2;
jac(:,6) = (2.*sin(beta).*(k.*cos(beta) - y.*cos(beta) - t.*sin(beta) + z.*sin(beta)))./(cos(beta).^2 + sin(beta).^2).^2 - (2.*cos(beta).*sin(theta).*(t.*cos(beta).*sin(theta) - z.*cos(beta).*sin(theta) + k.*sin(beta).*sin(theta) - y.*sin(beta).*sin(theta) + h.*cos(beta).^2.*cos(theta) + h.*sin(beta).^2.*cos(theta) - x.*cos(beta).^2.*cos(theta) - x.*sin(beta).^2.*cos(theta)))./((cos(beta).^2 + sin(beta).^2).^2.*(cos(theta).^2 + sin(theta).^2).^2);
if ~isempty(outliers_idx(outliers_idx==1))
    jac(outliers_idx==1,:) = zeros(size(jac(outliers_idx==1,:)));
end
end