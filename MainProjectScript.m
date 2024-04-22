%% Script initialization
addpath("tensor_toolbox-v3.6")


Real_Images_Dir = 'ISE789_images';
AI_Images_Dir = 'New_Images'; % Update with the actual path to directory 2
%% Remove black photos and resize the images in the directories to ensure same size
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

%%
Real_Image_mat = Images2Matrix(Real_Images_Dir);
AI_Image_mat = Images2Matrix(AI_Images_Dir);

%% Tucker decomposition
% Initialize cell arrays to store core tensors
num_images = size(Real_Image_mat, 3);
Real_image_tensor = cell(1, num_images);
AI_image_tensor = cell(1, num_images);

for rank = 1:150
% Loop through each image and perform Tucker decomposition
for i = 1:num_images
    % Perform Tucker decomposition on real image tensor
    Real_image_tensor{i} = tucker_als(tensor(double(Real_Image_mat(:,:,i))), [rank, rank]);
    
    % Perform Tucker decomposition on AI image tensor
    AI_image_tensor{i} = tucker_als(tensor(double(AI_Image_mat(:,:,i))), [rank, rank]);
end

%%
% Initialize an empty matrix to store the flattened tensors
num_samples = numel(Real_image_tensor);
core_size = numel(Real_image_tensor{1}.core.data); % Assuming all core tensors have the same size
flattened_cores = zeros(num_samples, core_size);

% Flatten each tensor and store it in the matrix
for i = 1:num_samples
    % Extract the core tensor from the Tucker decomposition result
    core_tensor = Real_image_tensor{i}.core;
    % Flatten the core tensor into a row vector
    flattened_core = reshape(core_tensor.data, 1, []);
    % Store the flattened core tensor in the matrix
    flattened_cores(i, :) = flattened_core;
end

% Initialize an empty matrix to store the flattened tensors
num_samples2 = numel(AI_image_tensor);
core_size2 = numel(AI_image_tensor{1}.core.data); % Assuming all core tensors have the same size
flattened_cores2 = zeros(num_samples2, core_size2);

% Flatten each tensor and store it in the matrix
for i = 1:num_samples2
    % Extract the core tensor from the Tucker decomposition result
    core_tensor2 = AI_image_tensor{i}.core;
    % Flatten the core tensor into a row vector
    flattened_core2 = reshape(core_tensor2.data, 1, []);
    % Store the flattened core tensor in the matrix
    flattened_cores2(i, :) = flattened_core2;
end

%%
% Split data into training and testing sets
data = vertcat(flattened_cores, flattened_cores2);
labels = [zeros(size(Real_Image_mat,3),1); ones(size(AI_Image_mat,3),1)];  % make labels

% Split data into training and testing sets
cv = cvpartition(labels, 'HoldOut', .2);
idxTrain = training(cv); % Index for training data
idxTest = test(cv); % Index for testing data

% Separate training and testing data
dataTrain = data(idxTrain,:);
labelsTrain = labels(idxTrain,:);
dataTest = data(idxTest,:);
labelsTest = labels(idxTest,:);
%%
% Create and train TreeBagger model
numTrees = 1000;
model = TreeBagger(numTrees, dataTrain, labelsTrain);

% Predict labels for test data
predictedLabels = predict(model, dataTest);

% Convert predicted labels to numeric format
predictedLabels = str2double(predictedLabels);

% Evaluate confusion matrix
% Compute evaluation metrics
accuracy(rank) = sum(predictedLabels == labelsTest) / numel(labelsTest);
confMat(:,:,rank) = confusionmat(labelsTest, predictedLabels);
precision(rank) = confMat(2,2,rank) / sum(confMat(:, 2,rank));
recall(rank) = confMat(2,2,rank) / sum(confMat(2, :,rank));
f1Score(rank) = 2 * (precision(rank) * recall(rank)) / (precision(rank) + recall(rank));
[X,Y,T,AUC] = perfcurve(labelsTest, predictedLabels, 1);

% disp(['Accuracy: ' num2str(accuracy)]);
% disp(['Precision: ' num2str(precision)]);
% disp(['Recall: ' num2str(recall)]);
% disp(['F1 Score: ' num2str(f1Score)]);
% disp(['AUC-ROC: ' num2str(AUC)]);
% disp('Confusion Matrix:');
% disp(confMat);
%%

% % Combine the datasets
% new_data = [Obama_cores; AIObama_cores];
% 
% % Create labels for the combined dataset
% num_real_images = size(Obama_cores, 1);
% num_ai_images = size(AIObama_cores, 1);
% new_labels = [ones(num_real_images, 1); zeros(num_ai_images, 1)];
% %%
% % Predict labels for the new data using the trained model
% predicted_labels_new = predict(model, new_data);
% 
% % Convert predicted labels to numeric format
% predicted_labels_new = str2double(predicted_labels_new);
% 
% % Evaluate confusion matrix
% confMat_new = confusionmat(new_labels, predicted_labels_new);
% 
% % Display confusion matrix
% disp('Confusion Matrix for New Data:');
% disp(confMat_new);
end
%%
figure;
subplot(2,2,1);
plot(accuracy);
subplot(2,2,1);
plot(Precision)






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
        processed_image = im_preprocessing(imread(imageName));
    
        % Display the processed image
        if i == 1
            figure;
            imshow(processed_image);
        end     
        % Store the processed image in imageMatrix
        imageMatrix(:, :, i) = processed_image;
    end    
    
    % Display the size of the resulting 3D matrix
    disp(['Size of image matrix: ' num2str(size(imageMatrix))]);
    
    % Display the first image from the matrix
    figure;
    imagesc(imageMatrix(:, :, 1));
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
    processed_image = edge(gray_im);
end

