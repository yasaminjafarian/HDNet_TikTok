function varargout = makeMovie(varargin)
% makeMovie M-file for makeMovie.fig
%      makeMovie by itself, creates a new makeMovie or raises the existing
%      singleton*.
%
%      H = makeMovie returns the handle to a new makeMovie or the handle to
%      the existing singleton*.
%
%      makeMovie('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in makeMovie.M with the given input arguments.
%
%      makeMovie('Property','Value',...) creates a new makeMovie or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before makeMovie_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to makeMovie_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help makeMovie

% Last Modified by GUIDE v2.5 27-Apr-2016 11:31:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
	'gui_Singleton',  gui_Singleton, ...
	'gui_OpeningFcn', @makeMovie_OpeningFcn, ...
	'gui_OutputFcn',  @makeMovie_OutputFcn, ...
	'gui_LayoutFcn',  [], ...
	'gui_Callback',   []);
if nargin && ischar(varargin{1})
	gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
	[varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
	gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before makeMovie is made visible.
function makeMovie_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to makeMovie (see VARARGIN)

% Choose default command line output for makeMovie
handles.output = true; % Just indicate that it ran successfully

%=====================================================================
% --- My Startup Code --------------------------------------------------
% Clear old stuff from console.
clc;
% Print informational message so you can look in the command window and see the order of program flow.
fprintf(1, 'Just entered makeMovie_OpeningFcn...\n');
% MATLAB QUIRK: Need to clear out any global variables you use anywhere
% otherwise it will remember their prior values from a prior running of the macro.
% (I'm not using any global variables in this program, so I commented it out.  Put it back in if you do put in and use any global variables.)
% clear global;
try	
	% Initialize some variables.
	handles.programFolder = cd; % Initialize
	handles.imageFolder = cd; % Initialize
	set(handles.figMainWindow, 'Visible', 'off');
	
	% Load up the initial values from the mat file.
	handles = LoadUserSettings(handles);
	
	% If the last-used image folder does not exist, but the imdemos folder exists, then point them to that MATLAB image demos folder instead.
	if exist(handles.imageFolder, 'dir') == 0
		% Folder stored in the mat file does not exist.  Try the imdemos folder instead.
		imdemosFolder = fileparts(which('cameraman.tif')); % Determine where demo images folder is (works with all versions of MATLAB).
		if exist(imdemosFolder, 'dir') == 0
			% imdemos folder exists.  Use it.
			handles.imageFolder = imdemosFolder;
		else
			% imdemos folder does not exist.  Use current folder.
			handles.imageFolder = cd;
		end
	end
	% handles.imageFolder will be a valid, existing folder by the time you get here.
	set(handles.txtFolder, 'string', handles.imageFolder);
	
	%msgboxh(handles.imageFolder);
	% Load list of images in the image folder.
	handles = LoadImageList(handles);
	% Select none of the items in the listbox.
	set(handles.lstImageList, 'value', []);
	% Update the number of images in the Analyze button caption.
	UpdateAnalyzeButtonCaption(handles);
	% Set up scrollbar captions.
	ScrollBarMoved(handles);
	
	% Load a splash image.
	%axes(handles.axesImage);
	fullSplashImageName = fullfile(handles.programFolder, '/Splash Images/Magic Hat.png');
	if exist(fullSplashImageName, 'file')
		% Display splash image.
		splashImage = imread(fullSplashImageName);
	else
		% Display something so it's not just blank.
		splashImage = peaks(300);
		fprintf('Warning: Splash image not found: %s\n', fullSplashImageName);
	end
	
	% Display image in the "axesImage" axes control on the user interface.
	hold off;	% IMPORTANT NOTE: hold needs to be off in order for the "fit" feature to work correctly.
	imshow(splashImage, 'InitialMagnification', 'fit', 'parent', handles.axesImage);
	axis off;
	
	txtInfo = sprintf('Movie Maker.');
	set(handles.txtInfo, 'string', txtInfo);
	
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
% Print informational message so you can look in the command window and see the order of program flow.
fprintf(1, 'Now leaving makeMovie_OpeningFcn.\n');

% Update handles structure
guidata(hObject, handles);
return; % from makeMovie_OpeningFcn()

% --- End of My Startup Code --------------------------------------------------
%=====================================================================


%=====================================================================
% --- Outputs from this function are returned to the command line.
function varargout = makeMovie_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
	% Print informational message so you can look in the command window and see the order of program flow.
	fprintf(1, 'Just entered makeMovie_OutputFcn...\n');
	% Get default command line output from handles structure
	varargout{1} = handles.output;
	
	% Maximize the window via undocumented Java call.
	% Reference: http://undocumentedmatlab.com/blog/minimize-maximize-figure-window
	MaximizeFigureWindow;
	
	% Print informational message so you can look in the command window and see the order of program flow.
	fprintf(1, 'Now leaving makeMovie_OutputFcn.\n');
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end

%=====================================================================
function LoadSplashImage(handles, handleToAxes)
try
	fullSplashImageName = fullfile(handles.programFolder, '/Splash Images/Splash.png');
	
	if exist(fullSplashImageName, 'file')
		% Display splash image.
		imgSplash = imread(fullSplashImageName);
	else
		% Display something
		imgSplash = peaks(300);
	end
	% Display image array in a window on the user interface.
	% Display in axes, storing handle of image for later quirk workaround.
	hold off;	% IMPORTANT NOTE: hold needs to be off in order for the "fit" feature to work correctly.
	imshow(imgSplash, 'InitialMagnification', 'fit', 'parent', handleToAxes);
	axis(handleToAxes, 'off');
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	WarnUser(errorMessage);
end
return; % from 	LoadSplashImage()

%=====================================================================
% --- Executes on clicking in lstImageList listbox.
% Display image from disk and plots histogram
function lstImageList_Callback(hObject, eventdata, handles)
% hObject    handle to lstImageList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: contents = get(hObject,'String') returns lstImageList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lstImageList
try
	% Change mouse pointer (cursor) to an hourglass.
	% QUIRK: use 'watch' and you'll actually get an hourglass not a watch.
	set(gcf,'Pointer','watch');
	drawnow;	% Cursor won't change right away unless you do this.
	
	% Update the number of images in the Analyze button caption.
	UpdateAnalyzeButtonCaption(handles);
	
	% Get image name
	selectedListboxItem = get(handles.lstImageList, 'value');
	if isempty(selectedListboxItem)
		% Bail out if nothing was selected.
		% Change mouse pointer (cursor) to an arrow.
		set(gcf,'Pointer','arrow');
		drawnow;	% Cursor won't change right away unless you do this.
		return;
	end
	% If more than one is selected, bail out.
	if length(selectedListboxItem) > 1
		baseImageFileName = '';
		% Change mouse pointer (cursor) to an arrow.
		set(gcf,'Pointer','arrow')
		drawnow;	% Cursor won't change right away unless you do this.
		return;
	end
	% If only one is selected, display it.
	set(handles.axesPlot, 'visible', 'off');	% Hide plot of results since there are no results yet.
	ListOfImageNames = get(handles.lstImageList, 'string');
	baseImageFileName = strcat(cell2mat(ListOfImageNames(selectedListboxItem)));
	fullImageFileName = [handles.imageFolder '/' baseImageFileName];	% Prepend folder.
	
	[folder, baseFileName, extension] = fileparts(fullImageFileName);
	switch lower(extension)
		case {'.mov', '.wmv', '.asf'}
			msgboxw('Mov and wmv format video files are not supported by MATLAB.');
			% Change mouse pointer (cursor) to an arrow.
			set(gcf,'Pointer','arrow');
			drawnow;	% Cursor won't change right away unless you do this.
			return;
			% 	case '.avi'
			% 		% The only video format supported natively by MATLAB is avi.
			% 		% A more complicated video player plug in is on MATLAB File Central
			% 		% that will support more types of video.  It has a bunch of DLL's and
			% 		% other files that you have to install.
			%
			% 		% Read the file into a MATLAB movie structure.
			% 		myVideo = aviread(fullImageFileName);
			% 		myVideoParameters = aviinfo(fullImageFileName);
			% 		numberOfFrames = myVideoParameters.NumFrames;
			%
			% 		% Extract a frame.
			% 		frameToView = uint8(floor(numberOfFrames/2));	% Take the middle frame.
			% 		imgFirstFrame = myVideo(frameToView).cdata;	% The index is the frame number.
			% 		cla(handles.axesPlot, 'reset');
			% 		imshow(imgFirstFrame, 'Parent', handles.axesPlot); % Display the first frame.
			%
			% 		% Play the movie in the axes.  It doesn't stretch to fit the axes.
			% 		% The macro will wait until it finishes before continuing.
			% 		axes(handles.axesImage);
			% 		hold off;
			% 		cla(handles.axesImage, 'reset'); % Let image resize, for example demo video rhinos.avi won't fill the image buffer if we don't do this.
			% 		movie(handles.axesImage, myVideo);
			% 		cla(handles.axesPlot, 'reset'); % Clear the mini-image from the plot axes.
			%
			% 	    guidata(hObject, handles);
			% 		% Change mouse pointer (cursor) to an arrow.
			% 		set(gcf,'Pointer','arrow');
			% 		drawnow;	% Cursor won't change right away unless you do this.
			% 		return;
		otherwise
			% Display the image.
			imgOriginal = DisplayImage(handles, fullImageFileName);
	end
	
	% If imgOriginal is empty (couldn't be read), just exit.
	if isempty(imgOriginal)
		% Change mouse pointer (cursor) to an arrow.
		set(gcf,'Pointer','arrow');
		drawnow;	% Cursor won't change right away unless you do this.
		return;
	end
		
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
% Change mouse pointer (cursor) to an arrow.
set(gcf,'Pointer','arrow');
drawnow;	% Cursor won't change right away unless you do this.
guidata(hObject, handles);
return % from lstImageList_Callback()

%=====================================================================
% Reads FullImageFileName from disk into the axesImage axes.
function imageArray = DisplayImage(handles, fullImageFileName)
% Read in image.
imageArray = []; % Initialize
try
	[imageArray, colorMap] = imread(fullImageFileName);
	% colorMap will have something for an indexed image (gray scale image with a stored colormap).
	% colorMap will be empty for a true color RGB image or a monochrome gray scale image.
catch ME
	% Will get here if imread() fails, like if the file is not an image file but a text file or Excel workbook or something.
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
	return;	% Skip the rest of this function
end

try
	% Display image array in a window on the user interface.
	%axes(handles.axesImage);
	hold off;	% IMPORTANT NOTE: hold needs to be off in order for the "fit" feature to work correctly.
	
	% Here we actually display the image in the "axesImage" axes.
	imshow(imageArray, 'InitialMagnification', 'fit', 'parent', handles.axesImage);
	
	% Display a title above the image.
	[folder, basefilename, extension] = fileparts(fullImageFileName);
	extension = lower(extension);
	% Display the title.
	caption = [basefilename, extension];
	title(handles.axesImage, caption, 'Interpreter', 'none', 'FontSize', 14);
	
	[rows, columns, numberOfColorChannels] = size(imageArray);
	% Get the file date
	fileInfo = dir(fullImageFileName);
	txtInfo = sprintf('%s\n\n%d lines (rows) vertically\n%d columns across\n%d color channels\n', ...
		[basefilename extension], rows, columns, numberOfColorChannels);
	% Tell user the type of image it is.
	if numberOfColorChannels == 3
		colorbar 'off';  % get rid of colorbar.
		txtInfo = sprintf('%s\nThis is a true color, RGB image.', txtInfo);
	elseif numberOfColorChannels == 1 && isempty(colorMap)
		colorbar 'off';  % get rid of colorbar.
		txtInfo = sprintf('%s\nThis is a gray scale image, with no stored color map.', txtInfo);
	elseif numberOfColorChannels == 1 && ~isempty(colorMap)
		txtInfo = sprintf('%s\nThis is an indexed image.  It has one "value" channel with a stored color map that is used to pseudocolor it.', txtInfo);
		colormap(colorMap);
		whos colorMap;
		%fprintf('About to apply the colormap...\n');
		% Thanks to Jeff Mather at the Mathworks to helping to get this working for an indexed image.
		colorbar('peer', handles.axesImage);
		%fprintf('Done applying the colormap.\n');
	end
	% Show the file time and date.
	txtInfo = sprintf('%s\n\n%s', txtInfo, fileInfo.date);
	set(handles.txtInfo, 'String', txtInfo);
	
	% Plot the histogram if requested.
	plotHistogram = get(handles.chkPlotHistograms, 'Value');
	if plotHistogram
		fprintf('About to plot histogram...\n');
		PlotImageHistogram(handles, imageArray);
		fprintf('Done plotting histogram.\n');
	end
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from DisplayImage


%=====================================================================
% Update Analyze button and tooltip string depending on how many files in the listbox were selected.
function UpdateAnalyzeButtonCaption(handles)
Selected = get(handles.lstImageList, 'value');
if length(Selected) > 1
	buttonCaption = sprintf('Step 5:  Go! Analyze %d images', length(Selected));
	set(handles.btnAnalyze, 'string', buttonCaption);
	set(handles.btnAnalyze, 'Tooltipstring', 'Display and analyze the selected image(s)');
elseif length(Selected) == 1
	set(handles.btnAnalyze, 'string', 'Step 5:  Go! Analyze 1 image');
	set(handles.btnAnalyze, 'Tooltipstring', 'Display and analyze the selected image');
else
	set(handles.btnAnalyze, 'string', 'Step 5:  Stop! Analyze no images');
	set(handles.btnAnalyze, 'Tooltipstring', 'Please select image(s) first');
end
return;

%=====================================================================
% --- Executes during object creation, after setting all properties.
function lstImageList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lstImageList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
	set(hObject,'BackgroundColor','white');
end
return

%=====================================================================
% --- Executes on clicking btnSelectFolder button.
% Asks user to select a directory and then loads up the listbox (via a call
% to LoadImageList)
function btnSelectFolder_Callback(hObject, eventdata, handles)
% hObject    handle to btnSelectFolder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%msgbox(handles.imageFolder);
try
	returnValue = uigetdir(handles.imageFolder,'Select folder');
	% returnValue will be 0 (a double) if they click cancel.
	% returnValue will be the path (a string) if they clicked OK.
	if returnValue ~= 0
		% Assign the value if they didn't click cancel.
		handles.imageFolder = returnValue;
		handles = LoadImageList(handles);
		set(handles.txtFolder, 'string' ,handles.imageFolder);
		guidata(hObject, handles);
		% Save the image folder in our ini file.
		SaveUserSettings(handles);
	end
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return

%=====================================================================
% --- Load up the listbox with tif files in folder handles.handles.imageFolder
function handles=LoadImageList(handles)
try
	ListOfImageNames = {};
	folder = handles.imageFolder;
	if ~isempty(handles.imageFolder)
		if exist(folder,'dir') == false
			warningMessage = sprintf('Note: the folder used when this program was last run:\n%s\ndoes not exist on this computer.\nPlease run Step 1 to select an image folder.', handles.imageFolder);
			msgboxw(warningMessage);
			return;
		end
	else
		msgboxw('No folder specified as input for function LoadImageList.');
		return;
	end
	% If it gets to here, the folder is good.
	ImageFiles = dir([handles.imageFolder '/*.*']);
	for Index = 1:length(ImageFiles)
		baseFileName = ImageFiles(Index).name;
		[folder, name, extension] = fileparts(baseFileName);
		extension = upper(extension);
		switch lower(extension)
			case {'.png', '.bmp', '.jpg', '.tif'}
				% Allow only PNG, TIF, JPG, or BMP images
				ListOfImageNames = [ListOfImageNames baseFileName];
			otherwise
		end
	end
	set(handles.lstImageList,'string',ListOfImageNames);
	% If the current listbox has fewer items in it than the value of the selected item in the last folder,
	% then the selected item will be off the end of the list, like you were looking at image 10 but the new folder has only 2 items in it.
	% In cases like that, the entire listbox will not show up.  So you have to make sure that the selected item is not more than the
	% number of items in the list.  We'll just deselect everything to make sure that doesn't happen.
	% Select none of the items in the listbox.
	set(handles.lstImageList, 'value', []);
	% Change caption to say Select All
	set(handles.btnSelectAllOrNone, 'string', 'Step 2:  Select All');
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from LoadImageList()

%=====================================================================
% --- Executes on clicking btnAnalyze button.
% Goes down through the list, displaying then analyzing each highlighted image file.
% Main processing is done by the function AnalyzeSingleImage()
function btnAnalyze_Callback(hObject, eventdata, handles)
% hObject    handle to btnAnalyze (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

try
	fontSize = 20;

	% Change mouse pointer (cursor) to an hourglass.
	% QUIRK: use 'watch' and you'll actually get an hourglass not a watch.
	set(gcf,'Pointer','watch');
	drawnow;	% Cursor won't change right away unless you do this.
	numberOfProcessedFiles = 0;
	
	% Then get list of all of the filenames in the list,
	% regardless of whether they are selected or not.
	ListOfImageNames = get(handles.lstImageList, 'string');
	numberOfFrames = length(ListOfImageNames);
	
	% Create a VideoWriter object to write the video out to a new, different file.
	outputFolder = handles.imageFolder;
	fullFileName = fullfile(outputFolder, 'TimeLapseMovie.avi');
	writerObj = VideoWriter(fullFileName);
	writerObj.FrameRate = round(handles.sldFrameRate.Value);
	open(writerObj);
	
	% Define the output size
	videoRows = 1080;
	vidColumns = 1980;
	startTime = tic;
	
	% Set up a way to bail out if they want to.
	handles.chkFinishNow.Visible ='on';
	handles.chkFinishNow.Value = 0;
	
	% Read the frames back in from disk, and convert them to a movie.
	% Preallocate constructedMovie, which will be an array of structures.
	% First get a cell array with all the frames.
	allTheFrames = cell(numberOfFrames,1);
	allTheFrames(:) = {zeros(videoRows, vidColumns, 3, 'uint8')};
	% Next get a cell array with all the colormaps.
	allTheColorMaps = cell(numberOfFrames,1);
	allTheColorMaps(:) = {zeros(256, 3)};
	% Now combine these to make the array of structures.
	constructedMovie = struct('cdata', allTheFrames, 'colormap', allTheColorMaps);
	axes(handles.axesImage);
	for frame = 1 : numberOfFrames
		% Construct an output image file name.
		sourceBaseFileName = ListOfImageNames{frame};
		sourceFrameFullFileName = fullfile(outputFolder, sourceBaseFileName);
		% Read the image in from disk.
		thisFrame = imread(sourceFrameFullFileName);
		% Resize it
		thisFrame = imresize(thisFrame, [videoRows, vidColumns]);
		cla;
		imshow(thisFrame);
		caption = sprintf('Frame %d of %d: %s', frame, numberOfFrames, sourceBaseFileName);
		title(caption, 'Interpreter', 'none', 'FontSize', fontSize);
		drawnow;
		% Convert the image into a "movie frame" structure.
		constructedMovie(frame) = im2frame(thisFrame);
		% Write this frame out to a new video file.
		if frame == 1
			firstFrameRepeat = str2double(handles.edtFirstFrameRepeat.String);
			% Write out the first frame this many times.  It might be the title slide.
			for k = 1 : firstFrameRepeat
				writeVideo(writerObj, thisFrame);
			end
		else
			% Write out frame just once.
			writeVideo(writerObj, thisFrame);
		end
		numberOfProcessedFiles = numberOfProcessedFiles + 1;
		
		elapsedSeconds = toc(startTime);
		handles.txtInfo.String = sprintf('Building Movie...\nOn frame %d of %d: %s\nElapsed time = %.1f seconds.', ...
			frame, numberOfFrames, sourceBaseFileName, elapsedSeconds);
		drawnow;
		% Check if they want to quit
		if handles.chkFinishNow.Value
			break;
		end
	end
	% Close the movie file.
	close(writerObj);
	% Reset way to bail out if they want to.
	handles.chkFinishNow.Visible ='off';
	handles.chkFinishNow.Value = 0;

	% Play the movie in the default program (like Windows Media Player).
	winopen(fullFileName);

% 	% Create new axes for our movie.
% 	figure;
% 	axis off;  % Turn off axes numbers.
% 	caption = sprintf('Movie built from all %d of the frames', numberOfFrames);
% 	title(caption, 'FontSize', fontSize);
% 	% Play the movie in the axes.
% 	movie(constructedMovie);
	% Note: if you want to display graphics or text in the overlay
	% as the movie plays back then you need to do it like I did at first
	% (at the top of this file where you extract and imshow a frame at a time.)
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end

set(gcf,'Pointer','arrow');
drawnow;	% Cursor won't change right away unless you do this.

guidata(hObject, handles);
return


%=====================================================================
% --- Executes on button press in btnSelectAllOrNone.
function btnSelectAllOrNone_Callback(hObject, eventdata, handles)
% hObject    handle to btnSelectAllOrNone (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
	% Find out button caption and take appropriate action.
	ButtonCaption = get(handles.btnSelectAllOrNone, 'string');
	if strcmp(ButtonCaption, 'Step 2:  Select All') == 1
		% Select all items in the listbox.
		% Need to find out how many items are in the listbox (both selected and
		% unselected).  It's quirky and inefficient but it's the only way I
		% know how to do it.
		% First get the whole damn listbox text into a cell array.
		caListboxString = get(handles.lstImageList, 'string');
		NumberOfItems = length(caListboxString);    % Get length of that cell array.
		AllIndices=1:NumberOfItems; % Make a vector of all indices.
		% Select all indices.
		set(handles.lstImageList, 'value', AllIndices);
		% Finally, change caption to say "Select None"
		set(handles.btnSelectAllOrNone, 'string', 'Step 2:  Select None');
		% It scrolls to the bottom of the list.  Use the following line
		% if you want the first item at the top of the list.
		set(handles.lstImageList, 'ListboxTop', 1);
	else
		% Select none of the items in the listbox.
		set(handles.lstImageList, 'value', []);
		% Change caption to say Select All
		set(handles.btnSelectAllOrNone, 'string', 'Step 2:  Select All');
	end
	% Update the number of images in the Analyze button caption.
	UpdateAnalyzeButtonCaption(handles);
	guidata(hObject, handles);
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end



%=====================================================================
% --- Executes on button press in chkPauseAfterImage.
function chkPauseAfterImage_Callback(hObject, eventdata, handles)
% hObject    handle to chkPauseAfterImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of chkPauseAfterImage
checkboxState = get(hObject,'Value');
if checkboxState
	set(handles.txtInfo, 'string', 'Now you will be able to inspect the results before it processes the next image.');
else
	set(handles.txtInfo, 'string', 'Now the image(s) will be analyzed without pausing for you will be able to inspect the results in between images.');
end


%=====================================================================
% Erases all lines from the image axes.  The current axes should be set first using the axes()
% command before this function is called, as it works from the current axes, gca.
function ClearLinesFromAxes(h)

%fprintf('ClearLinesFromAxes.1\n');
axesHandlesToChildObjects = findobj(h, 'Type', 'line');
%fprintf('ClearLinesFromAxes.2\n');
if ~isempty(axesHandlesToChildObjects)
	delete(axesHandlesToChildObjects);
end
return; % from ClearLinesFromAxes


%=====================================================================
% --- Executes during object creation, after setting all properties.
% EVEN THOUGH THIS FUNCTION IS EMPTY, DON'T DELETE IT OR ERRORS WILL OCCUR
function axesPlot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axesPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axesPlot


%=====================================================================
% --- Executes during object creation, after setting all properties.
% EVEN THOUGH THIS FUNCTION IS EMPTY, DON'T DELETE IT OR ERRORS WILL OCCUR
function axesImage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axesImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axesImage


%=====================================================================
% --- Executes during object creation, after setting all properties.
% EVEN THOUGH THIS FUNCTION IS EMPTY, DON'T DELETE IT OR ERRORS WILL OCCUR
function figMainWindow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figMainWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


%=====================================================================
% --- Executes on button press in btnExit.
function btnExit_Callback(hObject, eventdata, handles)
% hObject    handle to btnExit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
	% Print informational message so you can look in the command window and see the order of program flow.
	fprintf(1, 'Just entered btnExit_Callback...\n');
	
	% Save the current settings out to the .mat file.
	SaveUserSettings(handles);
	
	% Cause it to shutdown.
	delete(handles.figMainWindow);
	
	% Print informational message so you can look in the command window and see the order of program flow.
	fprintf(1, 'Now leaving btnExit_Callback.\n');
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end


%=====================================================================
function FillUpTable(handles)
try
	% Create sample data and row and column headers.
	columnHeaders = {'n', 'Result #1', 'Result #2'};
	tableData = cell(10, 3);
	for n = 1 : size(tableData, 1)
		rowHeaders{n} = sprintf('Row #%d', n);
		% Make up some data to put into the table control.
		tableData{n,1} = n;
		tableData{n,2} = 10*randi(9, 1,1);
		tableData{n,3} = sprintf('  Value = %.2f %s %.2f', rand(1,1), 177, rand(1));
	end
	
	% Apply the row and column headers.
	set(handles.uitable1, 'RowName', rowHeaders);
	set(handles.uitable1, 'ColumnName', columnHeaders);
	
	% Adjust columns widths.
	set(handles.uitable1, 'ColumnWidth', {30, 60, 180});
	
	% Display the table of values.
	set(handles.uitable1, 'data', tableData);
catch ME
	% Some error happened if you get here.
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return;  % AnalyzeSingleImage

%=====================================================================
% Takes histogram of 1-D array blobSizeArray, and plots it.
function [blobCounts, areaValues] = PlotHistogram(handles, blobSizeArray)
% Get a histogram of the blobSizeArray and display it in the histogram viewport.
numberOfBins = min([100 length(blobSizeArray)]);
[blobCounts, areaValues] = hist(blobSizeArray, numberOfBins);
% Plot the number of blobs with a certain area versus that area.
axes(handles.axesPlot);
bar(areaValues, blobCounts);
title('Histogram of Blob Sizes');
return;



% --- Executes when selected object is changed in grpRadButtonGroup.
function grpRadButtonGroup_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in grpRadButtonGroup
% eventdata  structure with the following fields (see UIBUTTONGROUP)
%	EventName: string 'SelectionChanged' (read only)
%	OldValue: handle of the previously selected object or empty if none was selected
%	NewValue: handle of the currently selected object
% handles    structure with handles and user data (see GUIDATA)
switch get(eventdata.NewValue, 'Tag') % Get Tag of selected object.
	case 'radOption1'
		% Code for when radiobutton1 is selected.
		txtInfo = sprintf('Option 1 is selected and the others are deselected.');
	case 'radOption2'
		% Code for when radiobutton2 is selected.
		txtInfo = sprintf('Option 2 is selected and the others are deselected.');
	case 'radOption3'
		% Code for when togglebutton1 is selected.
		txtInfo = sprintf('Option 3 is selected and the others are deselected.');
		% Continue with more cases as necessary.
	otherwise
		% Code for when there is no match.
end
set(handles.txtInfo, 'String', txtInfo);
return; % from grpRadButtonGroup_SelectionChangeFcn


%=============================================================================================
% --- Executes on slider movement.
function sldFrameRate_Callback(hObject, eventdata, handles)
% hObject    handle to sldFrameRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
% Set up the Scrollbar captions.
ScrollBarMoved(handles);


%=============================================================================================
% --- Executes during object creation, after setting all properties.
function sldFrameRate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sldFrameRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
	set(hObject,'BackgroundColor',[.9 .9 .9]);
end



%=============================================================================================
function ScrollBarMoved(handles)
try
	% Set up the Horizontal Scrollbar caption.
	scrollbarValue = get(handles.sldFrameRate,'Value');
	caption = sprintf('Frame Rate = %.2f', scrollbarValue);
	set(handles.txtFrameRate, 'string', caption);
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from SaveUserSettings()

%=============================================================================================
% --- Executes on mouse motion over figure - except title and menu.
function figMainWindow_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figMainWindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



%=============================================================================================
function edtFirstFrameRepeat_Callback(hObject, eventdata, handles)
% hObject    handle to edtFirstFrameRepeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edtFirstFrameRepeat as text
%        str2double(get(hObject,'String')) returns contents of edtFirstFrameRepeat as a double


%=============================================================================================
% --- Executes during object creation, after setting all properties.
function edtFirstFrameRepeat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edtFirstFrameRepeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
	set(hObject,'BackgroundColor','white');
end


%=============================================================================================
function mnuToolsSaveScreenshot_Callback(hObject, eventdata, handles)
% hObject    handle to mnuToolsSaveScreenshot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% IMPORTANT NOTE: YOU MUST HAVE DOWNLOADED import_fig()
% FROM THE FILE EXCHANGE FOR THIS FUNCTION TO WORK.
try
	% Save the figure
	% Have user browse for a file, from a specified "starting folder."
	% For convenience in browsing, set a starting folder from which to browse.
	startingFolder = 'C:\Program Files\MATLAB';
	if ~exist(startingFolder, 'dir')
		% If that folder doesn't exist, just start in the current folder.
		startingFolder = pwd;
	end
	% Get the name of the color image file that the user wants to save this figure into.
	defaultFileName = fullfile(startingFolder, 'MAGIC_Screenshot.png');
	[baseFileName, folder] = uiputfile(defaultFileName, 'Specify a file for your screenshot');
	if baseFileName == 0
		% User clicked the Cancel button.
		return;
	end
	screenshotFileName = fullfile(folder, baseFileName);
	% Display a message on the UI.
	txtInfo = sprintf('Please wait...\nSaving screenshot as\n%s', screenshotFileName);
	set(handles.txtInfo, 'String', txtInfo);
	% CALL export_fig, WHICH YOU MUST HAVE DOWNLOADED.
	export_fig(screenshotFileName);
	message = sprintf('Screenshot saved as:\n%s', screenshotFileName);
	set(handles.txtInfo, 'String', message);
	msgboxw(message);
catch ME
	% Some error happened if you get here.
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
	WarnUser('Are you sure you downloaded export_fig, and it is on the search path?');
end
return; % from mnuToolsSaveScreenshot_Callback


%=============================================================================================
function mnuToolsMontage_Callback(hObject, eventdata, handles)
% hObject    handle to mnuToolsMontage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
	% Change mouse pointer (cursor) to an hourglass.
	% QUIRK: use 'watch' and you'll actually get an hourglass not a watch.
	set(gcf,'Pointer','watch');
	% Get a list of all the filenames.
	ListOfImageNames = get(handles.lstImageList, 'string');
	% Get a list of what files they selected.
	selectedItems = get(handles.lstImageList, 'value');
	% If none are selected, use them all.
	numberOfSelectedImages = length(selectedItems);
	if numberOfSelectedImages <= 1
		numberOfSelectedImages = length(ListOfImageNames);
		selectedItems = 1 : numberOfSelectedImages;
	end
	caption = sprintf('Please wait...Constructing montage of %d images...', numberOfSelectedImages);
	title(caption, 'FontSize', 14);
	set(handles.txtInfo, 'string', caption);
	drawnow;	% Cursor won't change right away unless you do this.
	% Get a list of the selected files only.
	% Warning: This will not include folders so we will have to prepend the folder.
	ListOfImageNames = ListOfImageNames(selectedItems);
	for k = 1 : numberOfSelectedImages    % Loop though all selected indexes.
		% Get the filename for this selected index.
		baseImageFileName = cell2mat(ListOfImageNames(k));
		imageFullFileName = fullfile(handles.imageFolder, baseImageFileName);
		ListOfImageNames{k} = imageFullFileName;
	end
	% Figure out how many rows and there should be.
	% There is twice as much space horizontally as vertically so solve the equation 2*rows^2 = numberOfImages.
	rows = ceil(sqrt(numberOfSelectedImages/2));
	columns = 2 * rows;
	axes(handles.axesImage);
	cla(handles.axesImage, 'reset');
	handleToImage = handles.axesImage;
	hMontage = montage(ListOfImageNames, 'size', [rows columns]);
	caption = sprintf('Displaying a montage of %d images.', numberOfSelectedImages);
	title(caption, 'FontSize', 20);
	set(handles.txtInfo, 'string', caption);
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	set(handles.txtInfo, 'string', errorMessage);
	LoadSplashImage(handles, handles.axesImage);
	WarnUser(errorMessage);
end
% Change mouse pointer (cursor) to an arrow.
set(gcf,'Pointer','arrow');
drawnow;	% Cursor won't change right away unless you do this.

return; % from mnuToolsMontage_Callback


%=====================================================================
% Plots the histogram of grayImage in axes axesImage
% If grayImage is a double, it must be normalized between 0 and 1.
function [minGL, maxGL] = PlotImageHistogram(handles, imageArray)
try
	numberOfDimensions = ndims(imageArray);
	if numberOfDimensions == 1 || numberOfDimensions == 2
		% Grayscale 2D or 1D image.
		PlotGrayscaleHistogram(handles, imageArray)
	elseif numberOfDimensions == 3
		% True color RGB image.
		PlotRGBHistograms(handles, imageArray);
	else
		warningMessage = sprintf('Histograms are only supported for grayscale (2D) or color (3D) images');
		WarnUser(warningMessage);
	end
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % PlotImageHistogram

%=====================================================================
% Plots the histogram of grayImage in axes axesImage
% If grayImage is a double, it must be normalized between 0 and 1.
function [minGL, maxGL] = PlotGrayscaleHistogram(handles, grayImage)
try
	% Plot the histogram in the histogram viewport.
	axes(handles.axesPlot);  % makes existing axes handles.axesPlot the current axes.
	
	% Get the histogram
	[pixelCounts, grayLevels] = imhist(grayImage);
	
	% Plot it.
	bar(pixelCounts);
	title('Histogram of Gray Image', 'FontSize', 12);
	grid on;
	
	% Get the min and max GL for fun.
	minGL = min(grayImage(:));
	maxGL = max(grayImage(:));
	
	% Set up the x axis of the histogram to go from 0 to the max value allowed for this type of integer (uint8 or uint16).
	theClass = class(grayImage);
	% Only set xlim for integers.  For example the demo image blobs.png is logical not integer even though it's grayscale.
	if strfind(theClass, 'int')
		xlim([0 intmax(theClass)]);
	end
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % PlotGrayscaleHistogram

%=====================================================================
% Plots the histogram of imgColorImageArray in axes axesImage
% If imgArray is a double, it must be normalized between 0 and 1.
function [minGL, maxGL, gl1Percentile, gl99Percentile] = PlotRGBHistograms(handles, imgColorImageArray)
try
	% Get a histogram of the entire image.  But imhist only allows 2D images, not color ones.
	% First get individual channels from the original image.
	redBand = imgColorImageArray(:,:,1);
	greenBand = imgColorImageArray(:,:,2);
	blueBand = imgColorImageArray(:,:,3);
	
	% Use 256 bins.
	[redCounts, redGLs] = imhist(redBand, 256);    % make sure you label the axes after imhist because imhist will destroy them.
	[greenCounts, greenGLs] = imhist(greenBand, 256);    % make sure you label the axes after imhist because imhist will destroy them.
	[blueCounts, blueGLs] = imhist(blueBand, 256);    % make sure you label the axes after imhist because imhist will destroy them.
	
	% GLs goes from 0 (at element 1) to 255 (at element 256) but only some
	% these bins have data in them.  The upper ones may be 0.  Find the last
	% non-zero bin so we can plot just up to there to get better horizontal resolution.
	maxBinUsedR = find(redCounts, 1, 'last' );
	maxBinUsedG = find(greenCounts, 1, 'last' );
	maxBinUsedB = find(blueCounts, 1, 'last' );
	% If the entire image is zero, have the max bin be 0.
	if isempty(maxBinUsedR); maxBinUsedR = 1; end;
	if isempty(maxBinUsedG); maxBinUsedG = 1; end;
	if isempty(maxBinUsedB); maxBinUsedB = 1; end;
	% Take the largest one overall for plotting them all.
	maxBinUsed = max([maxBinUsedR maxBinUsedG maxBinUsedB]);
	
	% Get subportion of array that has non-zero data.
	redCounts = redCounts(1:maxBinUsed);
	greenCounts = greenCounts(1:maxBinUsed);
	blueCounts = blueCounts(1:maxBinUsed);
	GLs = redGLs(1:maxBinUsed);
	
	% Assign the max and min for the 3 color bands.
	minGL(1) = GLs(maxBinUsedR);
	maxGL(2) = GLs(maxBinUsedG);
	maxGL(3) = GLs(maxBinUsedB);
	
	% Calculate the 1% and 99% value of the CDF for the 3 color bands.
	gl1Percentile = ones(3, 1);	    % Preallocate one element for each color.
	gl99Percentile = ones(3, 1);		% Preallocate one element for each color.
	for color = 1:3
		switch color
			case 1
				counts = redCounts;
			case 2
				counts = greenCounts;
			case 3
				counts = blueCounts;
		end
		summed = sum(counts);
		cdf = 0;
		for bin = 1 : maxBinUsed
			cdf = cdf + counts(bin);
			if cdf < 0.01 * summed
				gl1Percentile(color) = GLs(bin);
			end
			if cdf > 0.99 * summed
				break;
			end
		end
		gl99Percentile(color) = GLs(bin);
	end
	
	% Plot the histogram in the histogram viewport.
	axes(handles.axesPlot);  % makes existing axes handles.axesPlot the current axes.
	% Plot the histogram as a line curve.
	plot(GLs, blueCounts, 'b', 'LineWidth', 3);
	hold on;
	plot(GLs, greenCounts, 'g', 'LineWidth', 3);
	plot(GLs, redCounts, 'r', 'LineWidth', 3);
	grid on;
	title('Red, Green, and Blue Histograms');
	maxCountInHistogram = max([redCounts(:); greenCounts(:); blueCounts(:)]);
	set(handles.axesPlot,'YLim',[0 maxCountInHistogram]);
	
	% Plot the sum of all of them
	% 		sumCounts = redCounts + greenCounts + blueCounts;
	% 		plot(GLs, sumCounts, 'k', 'LineWidth', 1);  % Plot sum in black color.
	% 		set(handles.axesPlot,'YLim',[0 max(sumCounts)]);
	
	% Set up custom tickmarks.
	set(handles.axesPlot,'XTick',      [0 20 40 60 80 100 120 140 160 180 200 220 240 255]);
	set(handles.axesPlot,'XTickLabel', {0 20 40 60 80 100 120 140 160 180 200 220 240 255});
	set(get(handles.axesPlot,'XLabel'),'string','Gray Level');
	ylabel('# of Pixels');
	xlabel('Gray Level');
	xlim([0 255]);
	hold off;
	
	axes(handles.axesImage);	% Switch current figure back to image box.
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % PlotRGBHistograms


%=============================================================================================
function mnuTools_Callback(hObject, eventdata, handles)
% hObject    handle to mnuTools (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%=============================================================================================
% Pops up a message box and waits for the user to click OK.
function msgboxw(in_strMessage)
uiwait(msgbox(in_strMessage));
return;

%=============================================================================================
% Pops up a help/information box and waits for the user to click OK.
function msgboxh(in_strMessage)
uiwait(helpdlg(in_strMessage));
return;

%==========================================================================================================================
% Warn user via the command window and a popup message.
function WarnUser(warningMessage)
fprintf(1, '%s\n', warningMessage);
uiwait(warndlg(warningMessage));
return; % from WarnUser()


% --- Executes on button press in chkPlotHistograms.
function chkPlotHistograms_Callback(hObject, eventdata, handles)
% hObject    handle to chkPlotHistograms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of chkPlotHistograms


% --- Executes on button press in radOption2.
function radOption2_Callback(hObject, eventdata, handles)
% hObject    handle to radOption2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of radOption2


% --- Executes on button press in radOption1.
function radOption1_Callback(hObject, eventdata, handles)
% hObject    handle to radOption1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of radOption1


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
try
	% Get the index (a number) and the item (the string in the drop down list) for the item they chose.
	contents = cellstr(get(handles.popupmenu1,'String')); % Returns popupmenu1 contents as cell array
	selectedIndex = get(handles.popupmenu1,'Value');	% Get the index of the item they selected: 1, 2, or 3.
	selectedItem = contents{selectedIndex};		% Returns selected item from popupmenu1
	% You can use the above code in any other callback to determine what item was selected.
	
	% Tell the user what they picked.
	message = sprintf('You selected item #%d from the popupmenu.\nThe string for that index number is : %s', selectedIndex, selectedItem);
	msgboxh(message);
	% Now you can add other code, if you want, to do other operations.
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from popupmenu1_Callback()


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
	set(hObject,'BackgroundColor','white');
end


%==================================================================================
% Save variables and GUI control settings so they can be recalled between sessions.
% I'm only saving the image folder and some of the settings here, but you could
% save all the settings of all the checkboxes, scrollbars, etc. on the GUI, if you want.
function SaveUserSettings(handles)
try
	matFullFileName = fullfile(handles.programFolder, [mfilename, '.mat']);
	% Save the current folder they're looking at.
	lastUsedImageFolder = handles.imageFolder;
	% Get current value of GUI controls, like checkboxes, etc.
	guiSettings.chkPauseAfterImage = get(handles.chkPauseAfterImage, 'Value');
	guiSettings.chkPlotHistograms = get(handles.chkPlotHistograms, 'Value');
	guiSettings.sldFrameRate = get(handles.sldFrameRate, 'Value');
	guiSettings.radOption1 = get(handles.radOption1, 'Value');
	guiSettings.radOption2 = get(handles.radOption2, 'Value');
	guiSettings.radOption3 = get(handles.radOption3, 'Value');

	% Save all the settings to a .mat file.
	save(matFullFileName, 'lastUsedImageFolder', 'guiSettings');
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from SaveUserSettings()


%=========================================================================================
% Recall variables from the last session, including the image folder and the GUI settings.
function handles = LoadUserSettings(handles)
try
	% Load up the initial values from the mat file.
	matFullFileName = fullfile(handles.programFolder, [mfilename, '.mat']);
	if exist(matFullFileName, 'file')
		% Pull out values and stuff them in structure initialValues.
		initialValues = load(matFullFileName);
		% Assign the image folder from the lastUsedImageFolder field of the structure.
		handles.imageFolder = initialValues.lastUsedImageFolder;
		
		% Get the last state of the controls.
		chkPauseAfterImage = initialValues.guiSettings.chkPauseAfterImage;
		chkPlotHistograms = initialValues.guiSettings.chkPlotHistograms;
		sldFrameRate = initialValues.guiSettings.sldFrameRate;
		radOption1 = initialValues.guiSettings.radOption1;
		radOption2 = initialValues.guiSettings.radOption2;
		radOption3 = initialValues.guiSettings.radOption3;
		
		% Send those recalled values to their respective controls on the GUI.
		set(handles.chkPauseAfterImage, 'Value', chkPauseAfterImage);
		set(handles.chkPlotHistograms, 'Value', chkPlotHistograms);
		set(handles.sldFrameRate, 'Value', sldFrameRate);
		set(handles.radOption1, 'Value', radOption1);
		set(handles.radOption2, 'Value', radOption2);
		set(handles.radOption3, 'Value', radOption3);
	else
		% If the mat file file does not exist yet, save the settings out to a new settings .mat file.
		SaveUserSettings(handles);
	end
catch ME
	errorMessage = sprintf('Error in program %s, function %s(), at line %d.\n\nError Message:\n%s', ...
		mfilename, ME.stack(1).name, ME.stack(1).line, ME.message);
	WarnUser(errorMessage);
end
return; % from LoadUserSettings()


% --- Executes on button press in chkFinishNow.
function chkFinishNow_Callback(hObject, eventdata, handles)
% hObject    handle to chkFinishNow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chkFinishNow
