function g = sigmoid(z)

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z 

g = 1./(1+exp(-z));

% =============================================================

end
