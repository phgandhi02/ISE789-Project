%% Script initialization
clear;clc;
close all;
addpath("C:\Users\pyrus\OneDrive - North Carolina State University\School\College\Senior\Spring 2024\ISE 789\tensor_toolbox-v3.6")


Real_Images_Dir = 'ISE789_images';
AI_Images_Dir = 'New_Images'; % Update with the actual path to directory 2
%% Resize the images in the directory
% Get a list of files in directory 1
Real_im_list = dir(fullfile(Real_Images_Dir, '*.jpg')); % Modify '*.jpg' based on your file extension

% Get a list of files in directory 2
AI_im_list = dir(fullfile(AI_Images_Dir, '*.jpg')); % Modify '*.jpg' based on your file extension

if size(imread(fullfile(Real_Images_Dir,Real_im_list(1).name))) ~= size(imread(fullfile(AI_Images_Dir,AI_im_list(1).name)))
    resize_images(Real_Images_Dir,[216 176]); 
    if any([Real_im_list.name] ~= [AI_im_list.name])
        disp('There is a difference in the image directories.')
        % Get the names of files in directory 2
        AI_listnames = {AI_im_list.name};
        % Loop through files in directory 1
        for i = 1:numel(Real_im_list)
            % Check if the current file in directory 1 exists in directory 2
            if ~ismember(Real_im_list(i).name, AI_listnames)
                % If the file doesn't exist in directory 2, delete it
                delete(fullfile(Real_Images_Dir, Real_im_list(i).name));
                disp(['Deleted: ' Real_im_list(i).name]);
            else
                % Read the image
                img = imread(fullfile(AI_Images_Dir, Real_im_list(i).name));
                
                % Check if the sum of the image array is zero to see if the
                % images returned as NSFW.
                if sum(img(:)) == 0
                    % If the sum is zero, delete the image
                    delete(fullfile(Real_Images_Dir, Real_im_list(i).name));
                    delete(fullfile(AI_Images_Dir, Real_im_list(i).name));
                    disp(['Deleted (Sum=0): ' Real_im_list(i).name]);
                end
            end
        end
    end
end


Real_Image_mat = Images2Matrix(Real_Images_Dir);
AI_Image_mat = Images2Matrix(AI_Images_Dir);

%% CP decomposition
% Loop through each image and add it to the matrix
i = 1;
imageName = fullfile(Real_Images_Dir, Real_im_list(i).name);
imageMatrix = imread(imageName);
for i=1:size(Real_Image_mat,3)
    Real_image_tensor = cp_als(tensor(double(Real_Image_mat(:,:,i))),2);
    AI_image_tensor = cp_als(tensor(double(AI_Image_mat(:,:,i))),2);
    Real_lambda(:,i) = Real_image_tensor.lambda;
    AI_lambda(:,i) = AI_image_tensor.lambda;
end

%%
vizopts = {'PlotCommands',{'bar','bar'},...
    'ModeTitles',{'Rows','Columns'},...
    'BottomSpace',.1,'HorzSpace',.04,'Normalize',0};
info1 = viz(Real_image_tensor,'Figure',1,vizopts{:});
%%
vizopts = {'PlotCommands',{'bar','bar'},...
    'ModeTitles',{'Rows','Columns'},...
    'BottomSpace',.1,'HorzSpace',.04,'Normalize',0};
info2 = viz(AI_image_tensor,'Figure',1,vizopts{:});
%%
figure;
plot(log(mean(Real_lambda,1)'),'b.'); hold on;
plot(log(mean(AI_lambda,1)'),'r');
figure;
plot(mean((AI_lambda - Real_lambda),1)','b.')
%%
% Split data into training and testing sets
data = [Real_lambda, AI_lambda];
labels = [zeros(size(Real_lambda,2),1); ones(size(AI_lambda,2),1)];  % make labels

% Split data into training and testing sets
cv = cvpartition(labels, 'HoldOut', .2);
idxTrain = training(cv); % Index for training data
dataTrain = data(idxTrain);
labelsTrain = labels(idxTrain,:);
idxTest = test(cv); % Index for testing data
dataTest = data(idxTest);
labelsTest = labels(idxTest,:);

% Create and train TreeBagger model
numTrees = 50;
model = TreeBagger(numTrees, dataTrain, labelsTrain);

% Predict labels for test data
predictedLabels = predict(model, dataTest);

% Convert predicted labels to numeric format
predictedLabels = str2double(predictedLabels);

% Evaluate confusion matrix
C = confusionmat(labelsTest, predictedLabels);
disp('Confusion Matrix:');
disp(C);

%%
Real_Image_vecs = tenmat(Real_Image_mat,1);
AI_Image_vecs = tenmat(AI_Image_mat,1);

lambda_new = zeros(size(image_vec,1),3);
err = zeros(size(estimation_set,1),1);
err_LLS = zeros(size(estimation_set,1),1);

design_matrix = khatrirao(set1_B,set1_A);

for im=1:size(estimation_set,1)
    lambda_new_LLS(im,:) = estimation_set(im,:)*design_matrix;
    err_LLS(im) = (norm(estimation_set(im,:) - ...
        (design_matrix*lambda_new_LLS(im,:)')','fro').^2)';
end

lambda_new_set2 = lambda_new_LLS(1:25,:);
lambda_new_setB = lambda_new_LLS(26:75,:);

opt_lambda_set2 = min(lambda_new_set2);


[muHat, sigmaHat] = normfit(lambda_new_set2);
threshold = zeros(size(muHat,2),2);
p = [.0167,.9];
for i=1:size(muHat,2)
    threshold(i,:) = icdf('Normal',p,muHat(i),sigmaHat(i));
end

threshold = threshold';

prediction = zeros(size(lambda_new_setB,1),1);

for i=1:size(lambda_new_setB)
    if any(lambda_new_setB(i,:) < threshold(1,:))
        prediction(i) = 1;
    elseif any(lambda_new_setB(i,:) > threshold(2,:))
        prediction(i) = 1;
    else
        prediction(i) = 0;
    end
end
true_SetB_labels = cat(1,zeros(9,1),ones(41,1));
C = confusionmat(prediction,true_SetB_labels);
disp(C);

%%
function resize_images(directory, image_size_arr)
    % List all JPEG files in the directory
    fileList = dir(fullfile(directory, '*.jpg'));
    
    % Loop through each file in the directory
    for i = 1:length(fileList)
        % Read the image
        filename = fullfile(directory, fileList(i).name);
        img = imread(filename);
        
        % Resize the image to 255x255 pixels
        img_resized = imresize(img, image_size_arr);
        
        % Save the resized image with the same filename
        imwrite(img_resized, filename);
    end
end

function imageMatrix = Images2Matrix(directory)
    % Get a list of all image files in the directory
    fileList = dir(fullfile(directory, '*.jpg')); % Change '*.jpg' to match your image file extension
    
    % Initialize a 3D matrix to store the images
    numImages = numel(fileList);
    firstImage = imread(fullfile(directory, fileList(1).name)); % Read the first image to get dimensions
    [m, n, ~] = size(firstImage);
    imageMatrix = zeros(m, n, numImages, 'uint8'); % Assuming 8-bit gray images
    
    % Loop through each image and add it to the matrix
    for i = 1:numImages
        imageName = fullfile(directory, fileList(i).name);
        imageMatrix(:, :, i) = im_preprocessing(imread(imageName));
    end
    
    % Display the size of the resulting 3D matrix
    disp(['Size of image matrix: ' num2str(size(imageMatrix))]);
    
    % Display the first image from the matrix
    %figure;
    %imshow(imageMatrix(:, :, 1));
end

function processed_image = im_preprocessing(image)
    % Define parameters for sharpness
    radius = 2;
    amount = 10;
    threshold = .2;
    % Apply sharpening 
    sharp_im = imsharpen(image, 'Radius', radius, 'Amount', amount,'Threshold', threshold);
    % Convert to grayscale
    gray_im = rgb2gray(sharp_im);
    % Apply edge detection
    %processed_image = edge(gray_im, 'log', 0.016);
    processed_image = sobel_operator(gray_im, 250);
end

function edge_image = sobel_operator(image,cutoff)
    im_gray = im2gray(image);
    [m,n] = size(im_gray); e = false(m,n);
    op = [-1 -2 -1;
          0 0 0;
          1 2 1]; 
    x_mask = op';  y_mask = op;
    fx = imfilter(im_gray,x_mask,'replicate'); 
    fy = imfilter(im_gray,y_mask,'replicate');
    f = fx.*fx+fy.*fy;
    edge_image = f;
    edge_image(f > cutoff) = 0;
end

