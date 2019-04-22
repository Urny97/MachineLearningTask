function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE ======================
% You need to finish the cost function and calculate the gradient result in
% here.

h_theta = sigmoid(X*theta);

J = (-1/m) * (y' * log(h_theta) + (1 - y)' * log(1 - h_theta)) + lambda/(2*m) * theta(2:end)'*theta(2:end);

theta(1) = 0;

grad = (X' * (h_theta - y))/m + lambda*theta/m;

% =============================================================

end
