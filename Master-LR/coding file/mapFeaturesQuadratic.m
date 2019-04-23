function out = mapFeaturesQuadratic(X, features)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to features used in the regularization exercise, which has the
%   same function of x2fx().
%   pair is the optimal feature pairs selected after obervation
%   degree is the degree of the polynomial features, not only quadratic
%
% NOTE: We couldn't find this function ourselves. We asked Ruben and Marijn
% how they did it.

out = ones(size(X(:,1)));

%aparte machten
for i = 1:2
    for n = features
        out(:, end+1) = X(:,features(n)).^i;
    end
end

%producten
for n = features
   for e = n+1 : features(end)
        out(:, end+1) = (X(:,features(n))).*(X(:,features(e))); 
    end
end