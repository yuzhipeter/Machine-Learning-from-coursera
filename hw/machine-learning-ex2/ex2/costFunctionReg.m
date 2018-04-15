function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% sigmoid=1./(1+exp(-X*theta));
% J=-1/m*sum(y.*log(sigmoid)+(1-y).*log(1-sigmoid));
% temp=(sigmoid-y);
% X_2=zeros(m);
% X_1=zeros(m);
% X_3=zeros(m);
% n=size(theta);
% %     J_partial=[0;0];
% %     J_partial(2)=sum(temp.*X_2)/m;
% %     J_partial(1)=sum(temp.*X_1)/m;
% for i=1:n(1);
%     grad(i)=sum(temp.*X(:,i))/m;
% end

sigmoid=1./(1+exp(-X*theta));
J=-1/m*sum(y.*log(sigmoid)+(1-y).*log(1-sigmoid))+lambda/2/m*(sum(theta.*theta)-theta(1)*theta(1));
temp=(sigmoid-y);
% X_2=zeros(m);
% X_1=zeros(m);
% X_3=zeros(m);
n=size(theta);
%     J_partial=[0;0];
%     J_partial(2)=sum(temp.*X_2)/m;
%     J_partial(1)=sum(temp.*X_1)/m;
for i=1:n(1);
    if i==1
        grad(i)=sum(temp.*X(:,i))/m;
    else
        grad(i)=sum(temp.*X(:,i))/m+lambda/m*theta(i);
    
end


% =============================================================

end
