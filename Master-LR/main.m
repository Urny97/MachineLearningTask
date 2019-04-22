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

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
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

% Set regularization parameter lambda to 0 and degree of polynomial to 1
lambda_lin= 0;
degree_lin = 1;

% Map features
trainingFeatures_lin = mapFeatures(trainingFeatures(:,1), trainingFeatures(:,2), degree_lin);
validationFeatures_lin = mapFeatures(validationFeatures(:,1), validationFeatures(:,2), degree_lin);

% Train theta
theta_lin = train(trainingFeatures_lin, trainingLabel, lambda_lin);

% F1-score of training set and validation set

p_train_lin = predict(theta_lin, trainingFeatures_lin);
F1_training_lin = F_Score(trainingLabel, p_train_lin);

p_val_lin = predict(theta_lin, validationFeatures_lin);
F1_validation_lin = F_Score(validationLabel, p_val_lin);

fprintf('F1 score for training set = %f \n', F1_training_lin)
fprintf('F1 score for validation set = %f \n', F1_validation_lin)

% Plot Boundary
plotDecisionBoundary(theta_lin, trainingFeatures_lin, trainingLabel, degree_lin);
hold on;
title(sprintf('Training: lambda = %g F1 = %g', lambda_lin, F1_training_lin))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Polynomial features from 2 features

degree_pol = 6;
trainingFeatures_pol = mapFeatures(trainingFeatures(:,1), trainingFeatures(:,2), degree_pol);
validationFeatures_pol = mapFeatures(validationFeatures(:,1), validationFeatures(:,2), degree_pol);

% Optimize lambda

lambda_pol = logspace(-10, 10, 500)

F_Score_train_pol = zeros(length(lambda_pol), 1);
F_Score_val_pol = zeros(length(lambda_pol), 1);

for i = 1:length(lambda_pol)
    theta_pol = train(trainingFeatures_pol, trainingLabel, lambda_pol(i));
    
    p_train_pol = predict(theta_pol, trainingFeatures_pol);
    F_Score_train_pol(i) = F_Score(trainingLabel, p_train_pol);
    
    p_val_pol = predict(theta_pol, validationFeatures_pol);
    F_Score_val_pol(i) = F_Score(validationLabel, p_val_pol);
end

figure;
plot(lambda_pol, F_Score_train_pol);
set(gca, 'XScale', 'log');
axis([10^(-10) 10^10 0 1]);
xlabel('lambda');
ylabel('F1 Score');
hold on
plot(lambda_pol, F_Score_val_pol);
legend('Training set','Validation set');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;