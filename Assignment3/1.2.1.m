%% Random data
clear, clc, clf
rng(2021)
X = randn(50,500);
mean_vector = mean(X,2);
X_zero_mean = X - mean_vector;
covariance_matrix = cov(X_zero_mean');
[V,D] = eigs(covariance_matrix, 50);
eigen_values = diag(D);
RMSD = zeros(49,1);

for i=1:49
    E = V(:, 1:i);
    z = E'*X_zero_mean;
    X_hat = E * z + mean_vector;
    RMSD(i) = sqrt(mean(mean((X-X_hat).^2)));
end

plot(RMSD)
xlabel('q')
ylabel('RMSD')
%% Correlated data
clear, clc, clf
load choles_all
mean_vector = mean(p,2);
p_zero_mean = p - mean_vector;
covariance_matrix = cov(p_zero_mean');
[V,D] = eigs(covariance_matrix, 21);
eigen_values = diag(D);
RMSD = zeros(20,1);

for i=1:20
    E = V(:, 1:i);
    z = E'*p_zero_mean;
    p_hat = E * z + mean_vector;
    RMSD(i) = sqrt(mean(mean((p-p_hat).^2)));
end

plot(RMSD)
xlabel('q')
ylabel('RMSD')