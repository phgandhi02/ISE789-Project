ai_im = double(im2gray(imread("AI Trump.jpg")));
real_im = double(im2gray(imread("Real Trump.jpg")));

edge_var = 1;

% K = [edge_var edge_var edge_var;
%         edge_var -1 edge_var;
%         edge_var edge_var edge_var];
% ai_im = imfilter(ai_im,K);
% real_im = imfilter(real_im,K);

figure;
imshow(sobel_operator(ai_im,100));

figure;
imshow(sobel_operator(real_im,100));

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