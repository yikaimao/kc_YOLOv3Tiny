% https://stackoverflow.com/questions/36761794/rearrange-sliding-blocks-into-columns-for-a-3d-array-im2col-in-3d-matlab
% https://stackoverflow.com/questions/25449279/efficient-implementation-of-im2col-and-col2im/25454746#25454746

function out = im2col_sliding(A,blocksize)

nrows = blocksize(1);
ncols = blocksize(2);

%// Get sizes for later usages
[m,n] = size(A);

%// Start indices for each block
start_ind = reshape(bsxfun(@plus,[1:m-nrows+1]',[0:n-ncols]*m),[],1); %//'

%// Row indices
lin_row = permute(bsxfun(@plus,start_ind,[0:nrows-1])',[1 3 2]);  %//'

%// Get linear indices based on row and col indices and get desired output
A = A';%change to row-major
out = A(reshape(bsxfun(@plus,lin_row,[0:ncols-1]*m),nrows*ncols,[]));

return;