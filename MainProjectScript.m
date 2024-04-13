% Define the directory containing the JPEG photos
directory = 'path/to/your/photos/directory';



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