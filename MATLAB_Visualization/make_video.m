clc
clear
close all
addpath('./utils/')


%% variables to change
path = '../test_data/'; % The test data path

name = '0043';  % the image name that you want to process.
% Note that you should have run HDNet_Inference.py before and have the
% predicted depth in [path,infer_out/,name,'.txt']

remove_images = true; 
% this code generates the images for each angle first then it generates the
% video. remove_images = true will remove the generated images. If you
% don not wish so set it as false.

%% making the video
tic
color_names = {'gray','color'};

for i = 1:2
    close all

    color_name = color_names{i};

    fprintf(['processing frame ', name , ' for ',color_name,' scale.\n']);


    d = dlmread([path,'infer_out/',name,'.txt'])-1.5;

    m = d>0;
    m = remove_n_boundary(m,2);


    C = imread([path,name,'_img.png']);

    C = C(:,:,1:3);

    [X,Y] = meshgrid(1:1:size(m,2),1:1:size(m,1));
    Y = Y * 1/size(m,1);
    X = X * 1/size(m,2);

    C1 = [eye(3),[0;0;0]];
    R1 = eye(3);
    K1 = eye(3); 
    K1(1,1)=1111.6; K1(2,2)=1111.6; K1(1,3)=960; K1(2,3)=540;
    origin1 = [838. , 48.];
    scaling1 = 3.5390625;
    [p,~,~] = Depth2Points3D_transformed(C1, R1, K1, d, m, origin1, scaling1);

    Z = d;
    Z(~m)=nan;
    Z(m)=p(:,3);

    if length(color_name) == 5
        if color_name == 'color'
            figure('visible','off')
            surf(Y-0.5,X-0.5,Z,C, ...   % Plot surface (flips rows of C, if needed)
                 'FaceColor', 'texturemap', ...
                 'EdgeColor', 'none')
        end
    else
        figure('visible','off')
        surfl(Y-0.5,X-0.5,Z,[0 -70]);
        shading flat; % Introduces shading
        colormap gray; % Use graylevel shading
    end

    grid off
    axis equal
    axis off
    set(gca,'XDir','reverse');
    xlim([-0.5 0.5])
    ylim([-0.5 0.5])

    OptionZ.FrameRate=30;OptionZ.Duration=5.5;OptionZ.Periodic=true;
    angl = 15;
    angl2 = 15;
    als1 = [-(91:0.2:90+angl),-(90+angl:-0.2:91)];
    als2 = [-(90:-0.2:90-angl2),-(90-angl2:0.2:90-1)];
    als = [als1,als2]';
    azs = [0*ones(size(als1)),0*ones(size(als2))]';
    mkdir([path,'infer_out/video/',name,'/',color_name,'/'])
    ims1 =CaptureFigVid([azs,als], [path,'infer_out/video/',name,'/',color_name,'/'],name,color_name,OptionZ);
end
delete TEST.jpg
%% make the combined gray and color images
d = dir([path,'infer_out/video/',name,'/','color','/*.jpg']);
mkdir([path,'infer_out/video/',name,'/','gray_color'])
for i =  1:length(d)
    im1 = imread([path,'infer_out/video/',name,'/','gray','/',int2str(i),'.jpg']);
    im2 = imread([path,'infer_out/video/',name,'/','color','/',int2str(i),'.jpg']);
    im = [im1,im2];
    imwrite(im,[path,'infer_out/video/',name,'/','gray_color/',int2str(i),'.jpg'])
end

%% convert the image sequence to video.
workingDir = [path,'infer_out/video/',name,'/','gray_color','/'];
out_dir = [path,'infer_out/video/',name,'/'];
imageNames = dir(fullfile(workingDir,'*.jpg'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter([out_dir,'video.avi']);
outputVideo.FrameRate = 30;
open(outputVideo)
for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,[int2str(ii),'.jpg']));
   writeVideo(outputVideo,img)
end
close(outputVideo)

%% remove the image files
if remove_images  
    rmdir([path,'infer_out/video/',name,'/','gray_color'],'s')
    rmdir([path,'infer_out/video/',name,'/','gray'],'s')
    rmdir([path,'infer_out/video/',name,'/','color'],'s')
end
toc