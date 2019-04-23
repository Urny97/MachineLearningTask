function [theta] = train(features, label, lambda)

% Initialize fitting parameters
initial_theta = zeros(size(features, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Learn theta
[theta, J] = fminunc(@(t)(costFunctionReg(t, features, label, lambda)), initial_theta, options);
