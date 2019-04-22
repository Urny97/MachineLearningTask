%% Clear the workspace, close all figures and load the data
clear;
close all;

load('Dataset\Features.mat');
load('Dataset\Label.mat');
load('Dataset\subject.mat');

%% Feature Selection
sittingLabel = double(label == 4);

gplotmatrix(features, features, sittingLabel);
title('Scatter plot of all features, sitting vs. rest')

chosenFeatures = features(:,[4 6]);

figure;
gplotmatrix(chosenFeatures(:,1), chosenFeatures(:,2), sittingLabel, 'kr', '*');
title('Scatter plot of feature 4 and 6, sitting vs. rest')
hold on;
xlabel('feat 4')
ylabel('feat 6')
%% Data seperation
% Normalisation
chosenFeatures = normalise(chosenFeatures);

% Create indices for data separation
trainingLastRow = round(0.4*length(features));
validationLastRow = trainingLastRow + round(0.3*length(features));
testLastRow  = validationLastRow + round(0.3*length(features)) - 1;

% Divide feature data in training, validation and test data
trainingFeatures = chosenFeatures(1:trainingLastRow, :);
validationFeatures = chosenFeatures(trainingLastRow:validationLastRow,:);
testFeatures = chosenFeatures(validationLastRow:testLastRow,:);

% Divide label data in training, validation and test data
trainingLabel = sittingLabel(1:trainingLastRow, :);
validationLabel = sittingLabel(trainingLastRow:validationLastRow,:);
testLabel = sittingLabel(validationLastRow:testLastRow,:);

% Divide subject data in training, validation and test data
trainingSubject = subject(1:trainingLastRow, :);
validationSubject = subject(trainingLastRow:validationLastRow,:);
testSubject = subject(validationLastRow:testLastRow,:);

clear trainingLastRow validationLastRow testLastRow;

%% Linear model with 2 features

% Setup data matrix
%[m, n] = size(trainingFeatures);
%trainingFeatures = [ones(m, 1) trainingFeatures];
degree = 1;
trainingFeatures = mapFeatures(trainingFeatures(:,1), trainingFeatures(:,2), degree);
% ^ werkt beide

% Set regularization parameter lambda to 0
lambda = 0;

% Train theta
[theta] = train(trainingFeatures, trainingLabel, lambda); 

% Plot Boundary
plotDecisionBoundary(theta, trainingFeatures, trainingLabel, degree);
hold on;
title(sprintf('lambda = %g', lambda))
%and F1 = %g', lambda, F_Score))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%% Validation

% Setup data matrix
%[m, n] = size(trainingFeatures);
%trainingFeatures = [ones(m, 1) trainingFeatures];
degree = 1;
validationFeatures = mapFeatures(validationFeatures(:,1), validationFeatures(:,2), degree);
% ^ werkt beide

% Set regularization parameter lambda to 0
lambda = 0;

% Train theta
[theta] = train(validationFeatures, validationLabel, lambda); 

% Plot Boundary
plotDecisionBoundary(theta, validationFeatures, validationLabel, degree);
hold on;
title(sprintf('validation: lambda = %g', lambda))
%and F1 = %g', lambda, F_Score))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%% F1-score

p = predict(theta, trainingFeatures);

F_Score = F_Score(trainingLabel, p);
fprintf('F-Score = %f \n', F_Score)

%% Polynomial features from 2 features

degree = 6;
trainingFeatures = mapFeatures(trainingFeatures(:,1), trainingFeatures(:,2), degree);

% Set regularization parameter lambda to 0
lambda = 0;

% Train theta
[theta] = train(trainingFeatures, trainingLabel, lambda); 

% Plot Boundary
plotDecisionBoundary(theta, trainingFeatures, trainingLabel, degree);
hold on;
title(sprintf('poly: lambda = %g', lambda))
%and F1 = %g', lambda, F_Score))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Optimize lambda

lambda_vec = [3^(-10):500: 3^(10)];

F_Score_train = zeros(length(lambda_vec), 1);
F_Score_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = train(trainingFeatures, trainingLabel, lambda);
    p_train = predict(theta, trainingFeatures);
    F_Score_train(i) = F_Score(trainingLabel, p_train);
    fprintf('F-Score = %f \n', F_Score_train(i))
    p_val = predict(theta, validationFeatures);
    F_Score_val(i) = F_Score(validationLabel, p_val);
    %error_train(i) = costFunctionReg(theta, trainingFeatures, trainingLabel, lambda);
    %error_val(i) = costFunctionReg(theta, validationFeatures, validationLabel, lambda);
end

figure;
plot(lambda_vec, F_Score_train, lambda_vec, F_Score_val);
hold on
title('Optimizing lambda')
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('F1 Score');

%% Validation

degree = 6;
validationFeatures = mapFeatures(validationFeatures(:,1), validationFeatures(:,2), degree);

% Set regularization parameter lambda to 0
lambda = 0;

% Train theta
[theta] = train(validationFeatures, validationLabel, lambda); 

% Plot Boundary
plotDecisionBoundary(theta, validationFeatures, validationLabel, degree);
hold on;
title(sprintf('validation poly: lambda = %g', lambda))
%and F1 = %g', lambda, F_Score))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;