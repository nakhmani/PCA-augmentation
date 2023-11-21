n = 100;

% Define the variables
S = ceil(0.7*201);  % total number of samples
B = ceil(0.7*98);   % number of benign samples
M = ceil(0.7*103);   % number of malignant samples

% Calculate c1 and c2
c1 = M / S;
c2 = B / S;

% Calculate nM and nB
nM = ceil(c1 * n)
nB = ceil(c2 * n)

Y = rand(nB, B);
[coeff,score,latent,tsquared,explained,mu] = pca(Y, 'NumComponents', nB);

% Define the number of top coefficients to select
num_top_coeffs = 13;

% Find the sum of absolute values of each coefficient, then select the top
% num_top_coeffs coefficients
coeff_sum_abs = sum(abs(coeff), 2);
[B, ind] = sort(coeff_sum_abs, 'descend');
B = B(1:num_top_coeffs);
ind = ind(1:num_top_coeffs);
result = score * coeff(ind,:)' + mu(ind)

% Check the size of the result
size(result)


%convert to csv
csvwrite('7030heart0-201-101.csv', result);