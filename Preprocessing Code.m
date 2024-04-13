a = imread("AI_Obama.jpg");
b = imread("Real Obama.jpg");
b = imcrop(b,[133.5 42.5 295 332]);
%obama [133.5 42.5 295 332]
%Shakira [137.5 7.5 220 218]

% Resize both images to 500x500
a_resized = imresize(a,[100,100]);
b_resized = imresize(b,[100,100]);

% Define parameters for sharpness
sharpness_radius = [2,10];
sharpness_amount = [5,50];
threshold_amount = [.2,.8];

% Loop through each combination of parameters
for radius = sharpness_radius
    for amount = sharpness_amount
        for threshold = threshold_amount
            % Part B: Apply sharpening
            sharp_a = imsharpen(a_resized, 'Radius', radius, 'Amount', amount);
            sharp_b = imsharpen(b_resized, 'Radius', radius, 'Amount', amount, 'Threshold', threshold);
            
            % Convert to grayscale
            gray_a = rgb2gray(sharp_a);
            gray_b = rgb2gray(sharp_b);
            
            % Part C: Apply edge detection
            e_a = edge(gray_a, 'log', 0.06);
            e_b = edge(gray_b, 'log', 0.06);
            
            % Create a new figure with maximum size
            fig = figure('units','normalized','outerposition',[0 0 1 1]);
            
            % Display both processed images
            subplot(1,2,1);
            imshow(e_a);
            title(sprintf('Image A: Radius=%d, Amount=%d,threshold=%d', radius, amount,threshold));
            
            subplot(1,2,2);
            imshow(e_b);
            title(sprintf('Image A: Radius=%d, Amount=%d,threshold=%d', radius, amount,threshold));
        end
    end
end