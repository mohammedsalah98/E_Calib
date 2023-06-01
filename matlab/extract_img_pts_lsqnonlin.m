function [img_pts_refined] = extract_img_pts_lsqnonlin(t_visualize,events_data, H, W, patternDims)

% Window events
window_events_idx = find(events_data(:,4) > t_visualize);
time_window_events = events_data(window_events_idx(1):window_events_idx(1)+4000, :);

% Delete polarities
time_window_events(:,3) = [];

% Normalize events
events_normalized = time_window_events(:,1:2);
events_normalized(:,1) = events_normalized(:,1) / W;
events_normalized(:,2) = events_normalized(:,2) / H;

if ~isempty(events_normalized)
    % DBSCAN
    idx = dbscan(events_normalized,0.015,10);
    time_window_events(idx==-1,:) = [];
    n_clusters = length(unique(idx));
    idx(idx==-1) = [];
    solutions = zeros(n_clusters, 3);
    residuals = zeros(n_clusters, 1);
    for i=1:n_clusters
        % Pick cluster (DBSCAN)
        events_cluster = time_window_events(idx==i, :);
        if ~isempty(events_cluster)
            events_cluster(:,3) = events_cluster(:,3) - events_cluster(1,3);
            events_cluster(:,3) = events_cluster(:,3) .* 1e4;
            % Define initial condition
            cluster_centroid = [mean(events_cluster(:,1)), mean(events_cluster(:,2))];
            target_cluster_offset = [events_cluster(:,1)-cluster_centroid(1), events_cluster(:,2)-cluster_centroid(2)];
            iwe_r = vecnorm(double(target_cluster_offset'));
            r = mean(iwe_r);
            z = mean(events_cluster(:,3));
            initial_condition = [r, cluster_centroid, 0, 0, z];
            events_optim = double(events_cluster);
            if ~anynan(events_optim)
                options = optimoptions(@lsqnonlin,'Algorithm', 'levenberg-marquardt', 'Display', 'off', 'SpecifyObjectiveGradient', true);
                [solution, error] = lsqnonlin(@(x) circle_cost_lsqnonlin(events_optim, x),initial_condition,[min(iwe_r),cluster_centroid(1)-r,cluster_centroid(2)-r,-pi/8, -pi/8, z], [max(iwe_r),cluster_centroid(1)+r,cluster_centroid(2)+r,pi/8, pi/8, z],options);
                solutions(i,:) = [solution(1), solution(2), solution(3)];
                residuals(i) = error;
            end
        end
    end
    % Remove obvious false detections
    solutions(residuals==0,:) = [];
    % Find refined image points
    img_calib = 255 .* ones(H,W);
    circles_detected = [solutions(:,2), solutions(:,3), 5.*ones(height(solutions),1)];
    img_calib = insertShape(img_calib,'FilledCircle',circles_detected, 'Color', 'black', 'Opacity', 1.0, 'SmoothEdges', false);
    imagePoints = detectCircleGridPoints(img_calib,patternDims,PatternType="asymmetric");
    if ~isempty(imagePoints)
        img_pts_refined = zeros(height(imagePoints), 2);
        for i=1:height(imagePoints)
            blob_imgPt = imagePoints(i,:);
            dist_to_fitted = vecnorm((solutions(:,2:3) - blob_imgPt)');
            [refined_pt, min_norm_idx] = min(dist_to_fitted);
            img_pts_refined(i,:) = solutions(min_norm_idx,2:3);
        end
        img_calib = insertShape(img_calib,'FilledCircle',[imagePoints, 5.*ones(height(imagePoints),1)], 'Color', 'red', 'Opacity', 1.0, 'SmoothEdges', false);
        imshow(img_calib)
    else
        img_pts_refined = [];
    end
else
    img_pts_refined = [];
end

end
