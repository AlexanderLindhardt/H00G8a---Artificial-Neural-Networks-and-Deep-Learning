
clear, clc, clf
%load('threes.mat')
load threes -ascii
%% Construct the mean three
clc, clf
colormap('gray')
three_mean = mean(threes, 1)
imagesc(reshape(three_mean,16,16),[0,1])

%% Plot eigen values
threes_zero_mean = threes - mean(threes, 2); 
covariance_matrix = cov(threes_zero_mean);
[V,D] = eigs(covariance_matrix, 256);
eigen_values = diag(D);
plot(eigen_values)
%% Recounstruct using fewer dimensions

