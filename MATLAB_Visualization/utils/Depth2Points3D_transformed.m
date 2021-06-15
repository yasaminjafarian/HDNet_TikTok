function [Points3D,rows,cols] = Depth2Points3D_transformed(cam_C, cam_R, cam_k, DepthMap, DepthMask, origin, scaling)
% make a copy of depth map
WorkOnDepthMap = DepthMap;

% replace the unavailable pixels from the mask to a dummy value in depth map
WorkOnDepthMap(~DepthMask)=1000;

% get the row and col of the pixels with available depth values
[rows,cols] = find(WorkOnDepthMap ~= 1000);

% get the available depth values
lambda = WorkOnDepthMap(WorkOnDepthMap~=1000);

% compute the camera center rom C matrix
cen = [-1*cam_C(1,4);-1*cam_C(2,4);-1*cam_C(3,4)];

% transform the xy locations from the scaled and cropped coordinate to the original coordinates
xy = [cols';rows']-1;
xy_translated = xy.* (scaling) + [ones(size(rows')).*origin(1);ones(size(rows')).*origin(2)];
xy_translated_scaled = xy_translated ;

% compute the 3D points from the original xy location
Points3D = ([lambda' ; lambda' ; lambda']  .* ( (cam_k*cam_R)\[xy_translated_scaled;ones(size(rows'))] ) ) + ...
    [cen(1)*ones(size(rows'));cen(2)*ones(size(rows'));cen(3)*ones(size(rows'))];
Points3D = Points3D';
end

