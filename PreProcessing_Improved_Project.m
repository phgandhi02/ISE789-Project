% Define folder path
folder_path = 'C:\Users\agmiran2\Pictures\Real_Images/';
%%
% Get a list of all image files in the folder
image_files = dir(fullfile(folder_path, '*.jpg'));

% Define parameters for sharpness
sharpness_radius = [2];
sharpness_amount = [10];
threshold_amount = [.2];

% Loop through each image
for i = 1:numel(image_files)
    % Read the image
    image_name = image_files(i).name;
    image_path = fullfile(folder_path, image_name);
    a = imread(image_path);
    figure
    imshow(a)
    
    % Resize the image
    a_resized = imresize(a, [150, 150]);
    
    % Loop through each combination of parameters
    for radius = sharpness_radius
        for amount = sharpness_amount
            for threshold = threshold_amount
                % Apply sharpening
                sharp_a = imsharpen(a_resized, 'Radius', radius, 'Amount', amount,'Threshold', threshold);
                
                % Convert to grayscale
                gray_a = rgb2gray(sharp_a);
                
                % Apply edge detection
                e_a = edge(gray_a, 'log', 0.016);
                figure
                imshow(e_a)
                % Save the edge-detected data as a CSV file
                csv_filename = fullfile(folder_path, sprintf('%s_sharpness_%d_amount_%d_threshold_%.2f.csv', image_name(1:end-4), radius, amount, threshold));
                csvwrite(csv_filename, e_a);
            end
        end
    end
end

%%
close all
