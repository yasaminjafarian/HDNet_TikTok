function [out_image_m,out_ref_points_m] = rotate_image( degree, in_image_m, in_ref_points_m )
%
% rotate_image - rotates an image given inside a matrix by the amount of "degree" counter-clockwise
%                using linear interpolation of the output grid points from the back-rotated input points
%                in this way, the output image will never have a "blank" point
%
% Format:   [out_image_m,out_ref_points_m] = rotate_image( degree, in_image_m, in_ref_points_m )
%
% Input:    degree          - rotation degree in dergees, counter-clockwise
%           in_image_m      - input image, given inside a matrix (gray level image only)
%           in_ref_points_m - points on the image wich their output coordinates will be given
%                             after the rotation. given format of this matrix is:
%                             [ x1,x2,...,xn;y1,y2,...,yn]
%
% Output:   out_image_m      - the output image
%           out_ref_points_m - the position of the input handle points after the rotation.
%                              this element is given in "in_ref_points_m" exists
%                              format of the matrix is the same as of "in_ref_points_m"
% 
% NOTE:     By definition of rotation, in order to perserve all the image inside the
%           rotated image space, the output image will be a matrix with a bigger size. 
%
%  NO INPUT ARGs - Launch demo and exit
if (nargin == 0)
    rotate_image_demo;
    out_image_m = [];
    return;
end
% check input
if ~exist('in_ref_points_m')
    in_ref_points_m = [];
end
% check for easy cases
switch (mod(degree,360))
case 0,     
    out_image_m      = in_image_m;
    out_ref_points_m = in_ref_points_m;
    return;
case 90,    
    out_image_m           = in_image_m(:,end:-1:1)';
    out_ref_points_m      = in_ref_points_m(end:-1:1,:);    
%     out_ref_points_m(2,:) = size(out_image_m,1) - out_ref_points_m(2,:);
    return;
case 180,   % TBD for rotation of the ref_points
    out_image_m           = in_image_m(end:-1:1,end:-1:1);
    out_ref_points_m      = in_ref_points_m;
    out_ref_points_m(2,:) = size(out_image_m,2) - out_ref_points_m(2,:);
    out_ref_points_m(1,:) = size(out_image_m,1) - out_ref_points_m(1,:);
    return;
case 270,   
    out_image_m           = in_image_m(end:-1:1,:)';
    out_ref_points_m      = in_ref_points_m(end:-1:1,:);
    out_ref_points_m(1,:) = size(out_image_m,2) - out_ref_points_m(1,:);
    return;
otherwise,  % enter the routine and do some calculations
end
% wrap input image by zeros from all sides
zeros_row    = zeros(1,size(in_image_m,2)+2);
zeros_column = zeros(size(in_image_m,1),1);
in_image_m   = [zeros_row; zeros_column,in_image_m,zeros_column; zeros_row ];
% build the rotation matrix
degree_rad = degree * pi / 180;
R = [ cos(degree_rad), sin(degree_rad); sin(-degree_rad) cos(degree_rad) ];
% input and output size of matrices (output size is found by rotation of 4 corners)
in_size_x       = size(in_image_m,2);
in_size_y       = size(in_image_m,1);
in_mid_x        = (in_size_x-1) / 2;
in_mid_y        = (in_size_y-1) / 2;
in_corners_m    = [ [0,0,in_size_x-1,in_size_x-1] - in_mid_x;
                    [0,in_size_y-1,in_size_y-1,0] - in_mid_y ];
out_corners_m   = R * in_corners_m;
% the grid (integer grid) of the output image and the output image
[out_x_r,out_y_r]   = rotated_grid( out_corners_m );
out_size_x          = max( out_x_r ) - min( out_x_r ) + 1;
out_size_y          = max( out_y_r ) - min( out_y_r ) + 1;
out_image_m         = zeros( ceil( out_size_y ),ceil( out_size_x ) );
out_points_span     = (out_x_r-min(out_x_r))*ceil(out_size_y) + out_y_r - min(out_y_r) + 1;
if ~isempty( in_ref_points_m )
    out_ref_points_m    = (R * [in_ref_points_m(1,:)-in_mid_x;in_ref_points_m(2,:)-in_mid_y]);
    out_ref_points_m    = [out_ref_points_m(1,:)-min( out_x_r )+1;out_ref_points_m(2,:)-min( out_y_r )+1];
else
    out_ref_points_m    = [];
end
    
% % for debug
% out_image_m(out_points_span) = 1;
% return;
% % end of for debug
% the position of points of the output grid in terms of the input grid
in_cords_dp_m   = inv(R) * [out_x_r;out_y_r];
x_span_left     = floor(in_cords_dp_m(1,:) + in_mid_x + 10*eps );
y_span_down     = floor(in_cords_dp_m(2,:) + in_mid_y + 10*eps );
x_span_right    = x_span_left + 1;
y_span_up       = y_span_down + 1;
dx_r            = in_cords_dp_m(1,:) - floor( in_cords_dp_m(1,:) + 10*eps );
dy_r            = in_cords_dp_m(2,:) - floor( in_cords_dp_m(2,:) + 10*eps );
point_span_0_0  = x_span_left*ceil(in_size_y)  + y_span_down + 1; % position of combined index in output matrix
point_span_1_0  = x_span_left*ceil(in_size_y)  + y_span_up + 1;
point_span_0_1  = x_span_right*ceil(in_size_y) + y_span_down + 1;
point_span_1_1  = x_span_right*ceil(in_size_y) + y_span_up + 1;
out_image_m(out_points_span) = ...
    in_image_m( point_span_0_0 ).*(1-dx_r).*(1-dy_r) + ...
    in_image_m( point_span_1_0 ).*(1-dx_r).*(  dy_r) + ...
    in_image_m( point_span_0_1 ).*(  dx_r).*(1-dy_r) + ...
    in_image_m( point_span_1_1 ).*(  dx_r).*(  dy_r);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              Inner function implementation                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_r,y_r] = rotated_grid( rect_points_m )
%
% rotated_grid - creates a grid of points bounded inside a rotated RECTANGLE
%
% Format:   [x_m,y_m] = rotated_grid( rect_points_m )
%
% Input:    rect_points_m   -   a set of (x;y) points which define a rectangle ordered clock-wise
%                               ( format: [x1,x2,x3,x4;y1,y2,y3,y4] )
%
% Output:   x_r,y_r         -   2 row vectors which hold the x and y positions of 
%                               the output grid
% 
% NOTE:     THE ASSUMPTION IS THAT THE RECTANGLE IS ORDERED CLOCK-WISE !!!
%           AND THAT THE GIVEN CO-ORDINATES ARE A RECTANGLE !
%
% make sure that the first point of the clock-wise-ordered rectange is of the most left point
[temp,idx] = min( rect_points_m(1,:) );
if ( idx > 1 )
    rect_points_m = [ rect_points_m(:,idx:end) , rect_points_m(:,1:idx-1) ];
end
% put into variables so it is easier to access/read the numbers
x1 = rect_points_m(1,1);
x2 = rect_points_m(1,2);
x3 = rect_points_m(1,3);
x4 = rect_points_m(1,4);
y1 = rect_points_m(2,1);
y2 = rect_points_m(2,2);
y3 = rect_points_m(2,3);
y4 = rect_points_m(2,4);
% initialization for grid creation
clipped_top     = floor( y2 );
clipped_bottom  = ceil( y4 );
fraction_bottom = clipped_bottom - y4;
rows            = ( clipped_top - clipped_bottom );
left_crossover  = y1 - y4;
right_crossover = y3 - y4;
% calculate the position of the edges (left and right) along the y axis
m = [0:rows] + fraction_bottom ;
switch (y1)
case y2, x_left = repmat( ceil( x4 ),size(m) );
case y4, x_left = repmat( ceil( x2 ),size(m) );
otherwise 
    x_left = ( m >= left_crossover ).*ceil( x2 - (x1-x2)/(y1-y2)*(rows-m+2*fraction_bottom) ) + ...
        ( m < left_crossover ).*ceil( x4 + (x1-x4)/(y1-y4)*m );
end
switch (y3)
case y2,    x_right = repmat( floor( x4 ),size(m) );
case y4,    x_right = repmat( floor( x2 ),size(m) );
otherwise
    x_right = ( m >= right_crossover ).*floor( x2 - (x3-x2)/(y3-y2)*(rows-m+2*fraction_bottom) ) + ...
        ( m < right_crossover ).*floor( x4 + (x3-x4)/(y3-y4)*m );
end
      
% build the output vectors (initialize)      
vec_length = sum(x_right-x_left+1);
x_r = zeros(1,vec_length );
y_r = zeros(1,vec_length );
% build the grid into the output vectors
cursor = 1;
for n = 1:length(m)
    if ( x_right(n) >= x_left(n) )
        span        = cursor:(x_right(n) - x_left(n) + cursor);
        x_r( span ) = x_left(n):x_right(n);
        y_r( span ) = m(n) + y4;
        cursor      = cursor + x_right(n) - x_left(n) + 1; 
    end
end
