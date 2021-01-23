% Yikai Mao 1/15/2021
% image padding for KiloCore 2

% Limitation: (implementation is based on YOLOv3Tiny)
% image must be a square matrix (image_width == image_height)
% padding type for convolution layer is 'SAME', '0', 'both'
% padding type for maxpool layer is 'SAME', 'replicate', 'post'

% WARNING:
% MATLAB use column-major layout!
% But this function is using row-major layout! (width * height)
% for easier C/C++ conversion (C/C++ use row-major layout)
% https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html

function [output, pad_size] = kc_padding(library, image, type, kernel_size)

pad_size = floor((kernel_size - 1) / 2);
image_size = size(image, 1);
channel_size = size(image, 3);

if type == "convolution"
    
    if pad_size == 0
        
        output = image;
    
    elseif library == "matlab"
        
        output = padarray(image, [pad_size pad_size]);
        
    elseif library == "kilocore"
        
        output = zeros(image_size + 2 * pad_size, ...
            image_size + 2 * pad_size, channel_size, 'single');
        
        for i = 1:channel_size
            output(1 + pad_size:size(output, 1) - pad_size,...
                1 + pad_size:size(output, 2) - pad_size, i) = image(:,:,i);
        end
        
    end
    
elseif type == "maxpool"
    
    if library == "matlab"
        
        output = padarray(image, [1,1], 'replicate', 'post');
        
    elseif library == "kilocore"
        
        output = zeros(image_size + 1, image_size + 1, ...
            channel_size, 'single');
        
        for i = 1:channel_size
            output(1:size(output, 1) - 1, 1:size(output, 2) - 1, i) = ...
                image(:,:,i);
            output(size(output, 1), 1:size(output, 2) - 1, i) = ...
                image(image_size, :, i);
            output(1:size(output, 1) - 1, size(output, 2), i) = ...
                image(:, image_size, i);
            output(size(output, 1), size(output, 2), i) = ...
                image(image_size, image_size, i);
        end
        
    end
    
end

end