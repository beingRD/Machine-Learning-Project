

addpath("Functions");
% Define the parameters for HOG computation
cell_size = 8;
num_bins = 9;

% Create an ImageDatastore object to read images from the CK+ dataset folder
imds = imageDatastore('Datasets/CK+', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Initialize an empty matrix to hold the extracted HOG features
hog_features = [];

% Loop over each image in the ImageDatastore
for i=1:981
    % Read the next image from the datastore
    img = imds.readimage(i);
[height, width, num_channels] = size(img);
num_cells_x = floor(width / cell_size);
num_cells_y = floor(height / cell_size);
    % Compute the HOG features for the current image
    hog =hog_f([8 8],9,img);
    hog_features_reshaped = reshape(hog, [num_cells_y, num_cells_x, num_bins]);

    % Append the HOG features for the current image to the matrix
    hog_features = [hog_features; hog_features_reshaped];
end
save('hog_f.mat',"hog_features");
