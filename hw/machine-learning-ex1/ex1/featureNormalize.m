function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
% size(X_norm);
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%
% a=mean(X(:,1));
% b=mean(X(:,2));
% mu=[a,b];
% sigma=[std(X(:,1)),std(X(:,2))];
% 
% % for i=1,length(sigma)
% %    X_norm(:,i)=(X_norm(:,i)-mu(i))/sigma(i); 
% % end
% % X_norm(:,1)=(X(:,1)-mu(1))/sigma(1);
% % X_norm(:,2)=(X(:,2)-mu(2))/sigma(2); 
% % mu
% % sigma
% % X_norm

n = size(X, 2);

for i = 1:n

	avg = mean(X(:, i));
	deviation = std(X(:, i));

	X_norm(:, i) = X_norm(:, i) - avg;
	X_norm(:, i) = X_norm(:, i) / deviation;

	mu(i) = avg;
	sigma(i) = deviation;

end
% mu
% sigma
% X_norm



% ============================================================

end
