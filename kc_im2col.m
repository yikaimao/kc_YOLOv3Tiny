% Yikai Mao 1/15/2021
% image to column vector for KiloCore 2

% Limitation: (implementation is based on YOLOv3Tiny)
% image must be a square matrix (image_width == image_height)
% kernel must be a square matrix (kernel_width == kernel_height)
% convolution stride must be 1 (shouldn't be too hard to change)
% https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
% https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster#im-2-col

% WARNING:
% MATLAB use column-major layout!
% But this function is using row-major layout! (width * height)
% for easier C/C++ conversion (C/C++ use row-major layout)
% https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html

function output = kc_im2col(library, image, kernel_size)

% image_size == image_width == image_height
image_size = size(image, 1);

% kernel_size == kernel_width == kernel_height
% (#rows)height =
% kernel_weight * kernel_height
% (#columns)width =
% (image_width - kernel_width + 1) * (image_height - kernel_height + 1)
output = zeros(kernel_size^2, (image_size-kernel_size+1)^2, 'single');

output_column_index = 1;

if library == "matlab"
    
    im2col(image, [kernel_size kernel_size]);
    
elseif library == "kilocore"
    
    for row = 1:image_size - (kernel_size - 1) % row
        for col = 1:image_size - (kernel_size - 1) % column
            
            % take the current sliding window
            window = image(row: row + kernel_size - 1, ...
                col: col + kernel_size - 1);
            
            % flatten the window then convert to column vector
            window = reshape(window',[],1);
            
            % equivalent:
            %window = reshape(window',[kernel_size^2,1]);
            
            % equivalent:
            %window = window';
            %window = window(:);
            
            % assign to output column
            output(:,output_column_index) = window;
            
            % move to next output column
            output_column_index = output_column_index + 1;
        end
    end
    
    % this is very fast, investigate
    %output = im2col_sliding(image, [kernel_size kernel_size]);
    
end

end