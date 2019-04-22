function out = mapFeatures(X1, X2, degree)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to features used in the regularization exercise, which has the
%   same function of x2fx().
%   pair is the optimal feature pairs selected after obervation
%   degree is the degree of the polynomial features, not only quadratic

out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end