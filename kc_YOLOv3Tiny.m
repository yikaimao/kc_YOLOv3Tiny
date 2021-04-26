% Yikai Mao 1/15 2021
% YOLOv3Tiny inference for KiloCore 2
% error calculation is for person.jpg only

%library = 'matlab';
library = 'kilocore';
quantize = 0; %takes ~35minutes!

load('weights_folded.mat');
load('bias_folded.mat');

% read input image
imgfile = 'images/person.jpg';
I = imread(imgfile);
I = im2single(I);
inputSz = 416;
input = letterbox_image(I, inputSz);

if quantize
    F = fimath();
    F.ProductMode = 'SpecifyPrecision';
    F.ProductWordLength = 16;
    F.ProductFractionLength = 11;
    F.SumMode = 'SpecifyPrecision';
    F.SumWordLength = 16;
    F.SumFractionLength = 11;
    for i=1:13
        weights_folded{i}=fi(weights_folded{i},1,16,11,F);
        bias_folded{i}=fi(bias_folded{i},1,16,11,F);
    end
    input = fi(input,1,16,11,F);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% YOLO CNN FEATURE EXTRACTOR

% convolution 1 -> maxpool 1
output1 = kc_convolution(library, input, weights_folded{1}, bias_folded{1}, 16, 3, 'padding', 'leaky');
maxpool1 = kc_maxpool(library, output1, 'no_padding');

load('data/max_pooling_1.mat');
[output_error_1, error_layer_1_mean, error_layer_1_max] = kc_error(max_pooling_1, maxpool1);

% convolution 2 -> maxpool 2
output2 = kc_convolution(library, maxpool1, weights_folded{2}, bias_folded{2}, 32, 3, 'padding', 'leaky');
maxpool2 = kc_maxpool(library, output2, 'no_padding');

load('data/max_pooling_2.mat');
[output_error_2, error_layer_2_mean, error_layer_2_max] = kc_error(max_pooling_2, maxpool2);

% convolution 3 -> maxpool 3
output3 = kc_convolution(library, maxpool2, weights_folded{3}, bias_folded{3}, 64, 3, 'padding', 'leaky');
maxpool3 = kc_maxpool(library, output3, 'no_padding');

load('data/max_pooling_3.mat');
[output_error_3, error_layer_3_mean, error_layer_3_max] = kc_error(max_pooling_3, maxpool3);

% convolution 4 -> maxpool 4
output4 = kc_convolution(library, maxpool3, weights_folded{4}, bias_folded{4}, 128, 3, 'padding', 'leaky');
maxpool4 = kc_maxpool(library, output4, 'no_padding');

load('data/max_pooling_4.mat');
[output_error_4, error_layer_4_mean, error_layer_4_max] = kc_error(max_pooling_4, maxpool4);

% convolution 5 -> maxpool 5
output5 = kc_convolution(library, maxpool4, weights_folded{5}, bias_folded{5}, 256, 3, 'padding', 'leaky');
maxpool5 = kc_maxpool(library, output5, 'no_padding');

load('data/max_pooling_5.mat');
[output_error_5, error_layer_5_mean, error_layer_5_max] = kc_error(max_pooling_5, maxpool5);

% convolution 6 -> maxpool 6
output6 = kc_convolution(library, maxpool5, weights_folded{6}, bias_folded{6}, 512, 3, 'padding', 'leaky');
maxpool6 = kc_maxpool(library, output6, 'padding');

load('data/max_pooling_6.mat');
[output_error_6, error_layer_6_mean, error_layer_6_max] = kc_error(max_pooling_6, maxpool6);

% convolution 7
output7 = kc_convolution(library, maxpool6, weights_folded{7}, bias_folded{7}, 1024, 3, 'padding', 'leaky');

load('data/leaky_relu_7.mat');
[output_error_7, error_layer_7_mean, error_layer_7_max] = kc_error(output7, leaky_relu_7);

% convolution 8
output8 = kc_convolution(library, output7, weights_folded{8}, bias_folded{8}, 256, 1, 'padding', 'leaky');

load('data/leaky_relu_8.mat');
[output_error_8, error_layer_8_mean, error_layer_8_max] = kc_error(output8, leaky_relu_8);

% convolution 9
output9 = kc_convolution(library, output8, weights_folded{9}, bias_folded{9}, 512, 3, 'padding', 'leaky');

load('data/leaky_relu_9.mat');
[output_error_9, error_layer_9_mean, error_layer_9_max] = kc_error(output9, leaky_relu_9);

% YOLO 1
output10 = kc_convolution(library, output9, weights_folded{10}, bias_folded{10}, 255, 1, 'padding', 'linear');

load('data/conv2d_10.mat');
[output_error_10, error_layer_10_mean, error_layer_10_max] = kc_error(output10, conv2d_10);

% route 1 (layers = -4)
% conv11 is taking input from conv8

% convolution 11
output11 = kc_convolution(library, output8, weights_folded{11}, bias_folded{11}, 128, 1, 'padding', 'leaky');

load('data/leaky_relu_11.mat');
[output_error_11, error_layer_11_mean, error_layer_11_max] = kc_error(output11, leaky_relu_11);

% upsample
upsample1 = kc_upsample(library, output11, 2);

load('data/upsampling_layer_1.mat');
[upsample_error_1, upsample_layer_1_mean, upsample_layer_1_max] = kc_error(upsampling_layer_1, upsample1);

% route 2 (layers = -1, 8)
route2 = cat(3, upsample1, output5);

load('data/routing_layer_2.mat');
[route_error_2, route_layer_2_mean, route_layer_2_max] = kc_error(routing_layer_2, route2);

% convolution 12
output12 = kc_convolution(library, route2, weights_folded{12}, bias_folded{12}, 256, 3, 'padding', 'leaky');

load('data/leaky_relu_12.mat');
[output_error_12, error_layer_12_mean, error_layer_12_max] = kc_error(output12, leaky_relu_12);

% YOLO 2
output13 = kc_convolution(library, output12, weights_folded{13}, bias_folded{13}, 255, 1, 'padding', 'linear');

load('data/conv2d_13.mat');
[output_error_13, error_layer_13_mean, error_layer_13_max] = kc_error(output13, conv2d_13);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% YOLO DETECTOR

confidenceThreshold = 0.25; % detection threshold, default = 0.25
overlapThreshold = 0.25; % NMS threshold, default = 0.25
load('data/mask.mat')
load('data/anchors.mat')
load('data/classNames.mat')

output10=reshape(output10, 13, 13, 85, 3);
output13=reshape(output13, 26, 26, 85, 3);

rawOutput = cell(2,1);
rawOutput{1} = output10;
rawOutput{2} = output13;

for i = 1:size(rawOutput, 1)
    % reshape 13x13x255 to 13x13x85x3, rows(y)*columns(x)*predictions*#anchorbox
    % 85 predictions on 3rd dim = 5 bounding box predictions + 80 class predictions
    % 5 bounding box predictions = x, y, width, height, Probability of object
    % 3 anchor boxes on 4th dim
    rawOutput{i} = reshape(rawOutput{i}, size(rawOutput{i}, 1), size(rawOutput{i}, 2), ...
        size(classNames, 1)+5, size(mask{i}, 2));
    predictions = permute(rawOutput{i},[1,2,4,3]);
    
    % apply activations
    % prepare cell position offsets, Cx/Cy in YOLO transform function
    outputSz = size(rawOutput{i},1);
    cellOffset = zeros(outputSz, outputSz, 3, 2);
    for j=1:3 % 3rd dim = 3 detections for each cell
        for k=1:outputSz % 1st dim = rows in Maltab = y direction in YOLO(reversed!)
            % 2nd dim = columns in Matlab = x direction in YOLO(reversed!)
            cellOffset(k,:,j,1) = (0:1:outputSz-1); % Cx in YOLO
            cellOffset(k,:,j,2) = zeros(1,outputSz)+k-1; % Cy in YOLO
        end
    end
    
    % bx = sigmoid(tx) + Cx
    % by = sigmoid(ty) + Cy
    predictions(:,:,:,1:2) = kc_sigmoid(predictions(:,:,:,1:2)) + cellOffset;
    
    % prepare anchor box priors, Pw/Py in YOLO transform function
    % rescale the anchor boxes and match its dimension to the detections
    anchorBox = anchors(mask{i},:)/(inputSz/outputSz); % might not work for other YOLO
    anchorBox = reshape(anchorBox, 1, 1, size(anchorBox,1), size(anchorBox,2));
    
    % bw = Pw * exp(tw), by = Py * exp(ty)
    predictions(:,:,:,3:4) = exp(predictions(:,:,:,3:4)).*anchorBox;
    
    % Pr(object) * IOU(b, object) = sigmoid(to)
    predictions(:,:,:,5:end) = kc_sigmoid(predictions(:,:,:,5:end));
    
    % calculates the absolute coordinates (x, y, w, h)
    predictions(:,:,:,1:4) = predictions(:,:,:,1:4)*(inputSz/outputSz);
    
    % Matlab's NMS function define x and y as the upper-left corner of the box
    % but YOLO define x and y as the center point of the box
    % so extra conversion is needed here
    predictions(:,:,:,1) = predictions(:,:,:,1)-predictions(:,:,:,3)/2; % x
    predictions(:,:,:,2) = predictions(:,:,:,2)-predictions(:,:,:,4)/2; % y
    
    % save the result as #detections*85
    predictions = reshape(predictions, size(mask{i}, 2)*outputSz^2, size(classNames, 1)+5);
    detections{i} = predictions;
end

% concatenate the detections from all YOLO layers
detections = cat(1, detections{:});

% filter out the low confidence detections
detections = detections(detections(:,5) > confidenceThreshold, :);

% select the best class prediction score
[scores, labels] = max(detections(:, 6:end), [], 2);

% probability of one class = class prediction .* objectness score
scores = scores.*detections(:,5);

% x, y, w, h
bboxes = detections(:,1:4);

% convert to text labels
labels = classNames(labels);

% NMS
if ~isempty(scores)
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(bboxes, scores, labels,...
        'RatioType','Union','OverlapThreshold', overlapThreshold); % Union/Min
    
    annotations = string(labels) + ": " + string(scores);
    output = insertObjectAnnotation(single(input), 'rectangle', bboxes, cellstr(annotations));
else
    % nothing detected
    output = input;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT RESULT

figure

% display original image
subplot(1,2,1);
imshow(single(input))
title('Original input image')

% display result
subplot(1,2,2);
imshow(output)
title('Detection result')
