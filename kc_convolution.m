% Yikai Mao 1/15/2021
% convolution layer for KiloCore 2

% Limitation: (implementation is based on YOLOv3Tiny)
% image must be a square matrix (image_width == image_height)
% kernel must be a square matrix (kernel_width == kernel_height)
% convolution stride must be 1 (shouldn't be too hard to change)

% WARNING:
% MATLAB use column-major layout!
% But this function is using row-major layout! (width * height)
% for easier C/C++ conversion (C/C++ use row-major layout)
% https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html

function output = kc_convolution(library, image, weights, bias, filters, kernel_size, pad, activation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dimenstions
image_size = size(image, 1);
image_depth = size(image, 3);
channel_size = filters;

% output_width = (image_width - kernel_width + 2 * padding) / stride + 1
% output_height = (image_height - kernel_height + 2 * padding) / stride + 1
output = zeros(image_size, image_size, channel_size, 'single');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% image padding
if pad == "padding"
    [image_padded, pad_size] = ...
        kc_padding(library, image, 'convolution', kernel_size);
else
    % never used in YOLOv3Tiny
    pad_size = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% vectorization
% flatten image
% (#rows)height =
% kernel_weight * kernel_height
% (#columns)width =
% (image_width - kernel_width + 1) * (image_height - kernel_height + 1)
image_vectorized = zeros(size(image_padded, 3) * (kernel_size^2), ...
    (image_size+2*pad_size-kernel_size+1)^2, 'single');

for i = 1:size(image_padded, 3)
    image_vectorized((kernel_size^2)*(i-1)+1:(kernel_size^2)*i, :) =...
        kc_im2col(library, image_padded(:,:,i), kernel_size);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% convolution
for channel = 1:channel_size
    
    % flatten weights
    weights_one_channel = weights(:,:,:,channel);
    weights_vectorized = zeros...
        (1, size(weights_one_channel(:), 1), 'single');
    
    if library == "matlab"
        
        weights_vectorized = weights_one_channel(:)';
        
    elseif library == "kilocore"
        
        weights_vectorized = reshape...
            (pagetranspose(weights_one_channel),1,[]);
        
    end
    
    bias_one_channel = bias(:,:,channel);
    
    if library == "matlab"
        
        output_one_channel = zeros(image_size, image_size, 'single');
        for i = 1:image_depth
            % conv2() is flipping the weights
            weight = flip(weights_one_channel(:,:,i));
            weight = flip(weight, 2);
            temp = conv2((image(:,:,i)), weight, 'same');
            output_one_channel = output_one_channel + temp;
        end
        output_one_channel = output_one_channel + bias_one_channel;
        
    elseif library == "kilocore"
        
        output_one_channel = ...
            (weights_vectorized * image_vectorized) + bias_one_channel;
        
        % dot product (slower, larger error, why?)
%         vec_weights = repmat(reshape(pagetranspose(weights_one_channel...
%             ),[],1),1,(image_size+2*pad_size-kernel_size+1)^2);
%         output_one_channel = ...
%             dot(vec_weights , image_vectorized) + bias_one_channel;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% activation
    if activation == "leaky"
        output_one_channel = max...
            (0, output_one_channel) + 0.1 .* min(0, output_one_channel);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% output result
    if library == "matlab"
        
        output_one_channel = reshape...
            (output_one_channel, [image_size image_size]);
        output(:,:,channel) = output_one_channel;
        
    elseif library == "kilocore"
        
        output_one_channel = reshape...
            (output_one_channel, [image_size image_size]);
        output_one_channel = output_one_channel';
        output(:,:,channel) = output_one_channel;
        
    end
    
end

end
