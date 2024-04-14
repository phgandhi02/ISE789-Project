%% Script initialization
clear;clc;
addpath("C:\Users\pyrus\OneDrive - North Carolina State University\School\College\Senior\Spring 2024\ISE 789\Project\ISE789-Project\tensor_toolbox-v3.6")


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
        fileNames2 = {AI_im_list.name};
        % Loop through files in directory 1
        for i = 1:numel(Real_im_list)
            % Check if the current file in directory 1 exists in directory 2
            if ~ismember(Real_im_list(i).name, fileNames2)
                % If the file doesn't exist in directory 2, delete it
                delete(fullfile(Real_Images_Dir, Real_im_list(i).name));
                disp(['Deleted: ' Real_im_list(i).name]);
            else
                % Read the image
                img = imread(fullfile(AI_Images_Dir, Real_im_list(i).name));
                
                % Check if the sum of the image array is zero
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

% Loop through each image and add it to the matrix
i = 1;
imageName = fullfile(Real_Images_Dir, fileList(i).name);
imageMatrix = imread(imageName);
image_tensor = cp_als(tensor(double(imageMatrix)),3);

%Real_Image_mat = Images2Matrix(Real_Images_Dir);
%AI_Image_mat = Images2Matrix(AI_Images_Dir);
%%
image_vecs = tenmat(Real_Image_mat,4);

image_vec1 = tenmat(double(set2),3); %correct because it's a vector for each image
image_vec2 = tenmat(double(setB),3); %correct because it's a vector for each image
lambda_new = zeros(size(image_vec1,1),3);
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
    imageMatrix = zeros(m, n, 3, numImages, 'uint8'); % Assuming 8-bit images
    
    % Loop through each image and add it to the matrix
    for i = 1:numImages
        imageName = fullfile(directory, fileList(i).name);
        imageMatrix(:, :, :, i) = imread(imageName);
    end
    
    % Display the size of the resulting 3D matrix
    disp(['Size of image matrix: ' num2str(size(imageMatrix))]);
    
    % Optionally, display the first image from the matrix
    figure;
    imshow(imageMatrix(:, :, :, 1));
end

