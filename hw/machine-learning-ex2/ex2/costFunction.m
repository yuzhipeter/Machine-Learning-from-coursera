function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
sigmoid=1./(1+exp(-X*theta));
J=-1/m*sum(y.*log(sigmoid)+(1-y).*log(1-sigmoid));
temp=(sigmoid-y);
X_2=zeros(m);
X_1=zeros(m);
X_3=zeros(m);
n=size(theta);
%     J_partial=[0;0];
%     J_partial(2)=sum(temp.*X_2)/m;
%     J_partial(1)=sum(temp.*X_1)/m;
for i=1:n(1);
    grad(i)=sum(temp.*X(:,i))/m;
end
% X_2=X(:,2);
% X_1=X(:,1);
% X_3=X(:,3);
% grad(1)=sum(temp.*X_1)/m;
% grad(2)=sum(temp.*X_2)/m;
% grad(3)=sum(temp.*X_3)/m;








% =============================================================

end
