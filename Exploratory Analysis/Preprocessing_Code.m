close all;

Person = 'Obama';
PATH = 'images\';
a = imread(append(PATH,'AI ',Person,'.jpg'));
b = imread(append(PATH,'Real ',Person,'.jpg'));
% b = imcrop(b,[133.5 42.5 295 332]); %obama
%b = imcrop(b,[137.5 7.5 220 218]); %Shakira

% Resize both images to 500x500
a_resized = imresize(a,[255,255]);
b_resized = imresize(b,[255,255]);

% Define parameters for sharpness
sharpness_radius = [4];
sharpness_amount = [50,10,80];
threshold_amount = [.8,.2];

%edge detection params
edge_thresh = 0.0036;

% Loop through each combination of parameters
for radius = sharpness_radius
    for amount = sharpness_amount
        for threshold = threshold_amount
            % Part B: Apply sharpening
            sharp_a = imsharpen(a_resized, 'Radius', radius, 'Amount', amount, 'Threshold', threshold);
            sharp_b = imsharpen(b_resized, 'Radius', radius, 'Amount', amount, 'Threshold', threshold);

            % Convert to grayscale
            gray_a = rgb2gray(a_resized);
            gray_b = rgb2gray(b_resized);
            
            % Part C: Apply edge detection
            e_a = edge(gray_a, 'log', edge_thresh);
            e_b = edge(gray_b, 'log', edge_thresh);
            
            % Create a new figure with maximum size
            fig = figure('units','normalized','outerposition',[0 0 1 1]);
            
            % Display both processed images
            subplot(1,2,1);
            imshow(e_a);
            title(sprintf('Image A: Radius=%d, Amount=%d,threshold=%d', radius, amount,threshold));
            
            subplot(1,2,2);
            imshow(e_b);
            title(sprintf('Image B: Radius=%d, Amount=%d,threshold=%d', radius, amount,threshold));
        end
    end
end