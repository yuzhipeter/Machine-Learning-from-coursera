function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp=X*theta-y;
    X_2=X(:,2);
    X_1=X(:,1);
    J_partial=[0;0];
    J_partial(2)=sum(temp.*X_2)/m;
    J_partial(1)=sum(temp.*X_1)/m;
%     for i=1:length(y)
%         J_partial(1)=J_partial(1)+2*(temp(i))/m;
%         J_partial(2)=J_partial(2)+2*(temp(i)*X_2(i))/m;
%     end
    theta=theta-alpha*J_partial;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
