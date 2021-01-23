% Yikai Mao 1/19/2021
% upsample layer for KiloCore 2

% Limitation: (implementation is based on YOLOv3Tiny)
% image must be a square matrix (image_width == image_height)
% kernel must be a square matrix (kernel_width == kernel_height)

% WARNING:
% MATLAB use column-major layout!

function output = kc_upsample(library, image, stride)

image_size = size(image, 1);
channel_size = size(image, 3);

if library == "matlab"
    
    output = repelem(image, stride, stride);
    
elseif library == "kilocore"
    
    output = zeros(image_size * stride, ...
        image_size * stride, channel_size, 'single');
    
    % simple upsample, just copy the elements
    for i = 1:channel_size
        for j = 1:image_size * stride
            for k = 1:image_size * stride
                output(j, k, i) = ...
                    image(ceil(j / stride), ceil(k / stride), i);
            end
        end
    end
    
end

end