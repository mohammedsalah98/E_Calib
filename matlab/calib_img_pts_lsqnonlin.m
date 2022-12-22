clear
close all
clc
warning off all

% Read h5 file
file_dir = input('Specify h5 data sequence directory: ');
events_data = h5read(file_dir, '/events_data');

% Sensor resolution
H = input('Specify image height: ');
W = input('Specify image width: ');

% Pattern dimensions
pattern_height = input('Specify pattern height: ');
pattern_width = input('Specify pattern width: ');
patternDims = [pattern_height pattern_width];

% Diagonal spacing
diagonal_spacing = input('Specify pattern diagonal spacing: ');

% Fix format
events_data = events_data';

% Sim Time
t_sample = 1/30.0;
events_data(:,4) = (events_data(:,4) - events_data(1,4));
max_time = floor(events_data(end,4));
t_array = linspace(5,max_time-5,round(max_time/t_sample));

% Main
img_pts_array = [];
count = 0;
for i=1:length(t_array)
    img_pts_current = extract_img_pts_lsqnonlin(t_array(i), events_data, H, W, patternDims);
    if ~isempty(img_pts_current)
        img_pts_array = cat(3,img_pts_array, img_pts_current);
    end
    t_current = t_array(i);
end

% Calibration for all images
worldPoints = generateCircleGridPoints(patternDims,diagonal_spacing);
imageSize = [H, W];
params = estimateCameraParameters(img_pts_array,worldPoints, ...
                                  'ImageSize',imageSize, 'EstimateSkew', true, 'NumRadialDistortionCoefficients',2,'EstimateTangentialDistortion',false);

% Find calibration outliers and recalibrate
total_repr_errors = zeros(size(params.ReprojectionErrors,3),1);
for i=1:size(params.ReprojectionErrors,3)
    total_repr_errors(i) = mean(vecnorm(vecnorm(params.ReprojectionErrors(:,:,i), 2, 2)));
end

outliers_calibration = isoutlier(total_repr_errors, 'quartiles');
img_pts_array(:,:,outliers_calibration==1) = [];
params = estimateCameraParameters(img_pts_array,worldPoints, ...
                                  'ImageSize',imageSize, 'EstimateSkew', true, 'NumRadialDistortionCoefficients',3,'EstimateTangentialDistortion',true);
showReprojectionErrors(params);
