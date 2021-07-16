
clear, clc, clf
%load('threes.mat')
load threes -ascii
%% Construct the mean three
clc, clf
colormap('gray')
three_mean = mean(threes, 1)
imagesc(reshape(three_mean,16,16),[0,1])

%% Plot eigen values
mean_vector = mean(threes, 2)';
threes_zero_mean = threes' - mean_vector; 
covariance_matrix = cov(threes_zero_mean');
[V,D] = eigs(covariance_matrix, 256);
eigen_values = diag(D);
plot(eigen_values)
xlabel('eigenvalue')
ylabel('')
%% Recounstruct using fewer dimensions and plot
clc, clf
for i=1:4
    E = V(:, 1:i);
    z = E'*threes_zero_mean;
    threes_hat = E * z + mean_vector;
    hold on
    
    subplot(4,4,i)
    imagesc(reshape(threes_hat(:,1),16,16),[0,1])
    subplot(4,4,i+4)
    imagesc(reshape(threes_hat(:,2),16,16),[0,1])
    subplot(4,4,i+8)
    imagesc(reshape(threes_hat(:,3),16,16),[0,1])
    subplot(4,4,i+12)
    imagesc(reshape(threes_hat(:,6),16,16),[0,1])
    
end
%%
clc, clf
RMSD = zeros(50,1);
for i=1:50
    E = V(:, 1:i);
    z = E'*threes_zero_mean;
    threes_hat = E * z + mean_vector;
    RMSD(i) = sqrt(mean(mean((threes'-threes_hat).^2)));
end

plot(RMSD)
xlabel('q')
ylabel('RMSD')
hold on
%% cumsum

eigen_cumsum = cumsum(eigen_values,'reverse');
plot(eigen_cumsum(2:51)/sum(eigen_values))