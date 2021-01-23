% Yikai Mao 1/19/2021
% maxpool layer for KiloCore 2

% Limitation: (implementation is based on YOLOv3Tiny)
% image must be a square matrix (image_width == image_height)
% kernel must be a square matrix (kernel_width == kernel_height)

% WARNING:
% MATLAB use column-major layout!

% https://www.mathworks.com/matlabcentral/answers/409032-how-do-i-compute-the-maxpool-of-a-image-let-us-say-stride-of-2-2-on-a-mxn-matrix

function output = kc_maxpool(library, image, padding)

image_size = size(image, 1);
channel_size = size(image, 3);

% padding only for maxpool layer 6 with stride = 1
% weird padding, add one column to the right
% and one row to the bottom, with padding value = boarder values
% equivalent: padarray(image, [1,1], 'replicate', 'post');
if padding == "padding"
    output = zeros(image_size, image_size, channel_size, 'single');
    [image, ~] = kc_padding(library, image, 'maxpool', 2);
    stride = 1;
    
    % maxpooling
    for i = 1:channel_size
        im_nw = image(1:stride:image_size, 1:stride:image_size, i);
        im_sw = image(2:stride:image_size+1, 1:stride:image_size, i);
        im_se = image(2:stride:image_size+1, 2:stride:image_size+1, i);
        im_ne = image(1:stride:image_size, 2:stride:image_size+1, i);
        output(:,:,i) = max(cat(3,im_nw,im_sw,im_se,im_ne),[],3);
    end
    
    % no padding
else
    output = zeros(image_size / 2, image_size / 2, channel_size, 'single');
    stride = 2;
    
    % maxpooling
    for i = 1:channel_size
        im_nw = image(1:stride:image_size, 1:stride:image_size, i);
        im_sw = image(2:stride:image_size, 1:stride:image_size, i);
        im_se = image(2:stride:image_size, 2:stride:image_size, i);
        im_ne = image(1:stride:image_size, 2:stride:image_size, i);
        output(:,:,i) = max(cat(3,im_nw,im_sw,im_se,im_ne),[],3);
    end
    
end

end