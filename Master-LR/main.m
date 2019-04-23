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
close all

% Normalisation
featuresNorm = normalise(features); 
chosenFeatures = featuresNorm(:,[4 6]);

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

% Plot Boundary
plotDecisionBoundary(theta_lin, trainingFeatures_lin, trainingLabel, degree_lin);
hold on;
title(sprintf('Training: lambda = %g F1 = %g', lambda_lin, F1_training_lin))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('F1 score for training set = %.0001f \n', F1_training_lin)
fprintf('F1 score for validation set = %.0001f \n', F1_validation_lin)

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Polynomial features from 2 features
close all

degree_pol = 6;
trainingFeatures_pol = mapFeatures(trainingFeatures(:,1), trainingFeatures(:,2), degree_pol);
validationFeatures_pol = mapFeatures(validationFeatures(:,1), validationFeatures(:,2), degree_pol);

% Optimize lambda

lambda_pol = logspace(-10, 10, 250);

F_Score_train_pol = zeros(length(lambda_pol), 1);
F_Score_val_pol = zeros(length(lambda_pol), 1);

for i = 1:length(lambda_pol)
    theta_pol = train(trainingFeatures_pol, trainingLabel, lambda_pol(i));
    
    p_train_pol = predict(theta_pol, trainingFeatures_pol);
    F_Score_train_pol(i) = F_Score(trainingLabel, p_train_pol);
    
    p_val_pol = predict(theta_pol, validationFeatures_pol);
    F_Score_val_pol(i) = F_Score(validationLabel, p_val_pol);
end

% plot f1 score vs lambda
figure;
plot(lambda_pol, F_Score_train_pol);
set(gca, 'XScale', 'log');
axis([10^(-10) 10^10 0 1]);
xlabel('lambda');
ylabel('F1 Score');
hold on
plot(lambda_pol, F_Score_val_pol);
legend('Training set','Validation set');
title('Using Features 4 & 6');

% plot decision boundary for validation set with highest F1 score
index = find(F_Score_val_pol == max(F_Score_val_pol));
best_lambda_val = lambda_pol(index);
best_theta_val = train(trainingFeatures_pol, trainingLabel, best_lambda_val);

plotDecisionBoundary(best_theta_val, validationFeatures_pol, validationLabel, degree_pol);
hold on;
title(sprintf('Validation: lambda = %g F1 = %g F1 training = %g', best_lambda_val, F_Score_val_pol(index), F_Score_train_pol(index)))
xlabel('feat 4')
ylabel('feat 6')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Linear classifier with all 8 features
close all

% Divide feature data in training, validation and test data
newTrainingFeatures = featuresNorm(1:trainingLastRow, :);
newValidationFeatures = featuresNorm(trainingLastRow:validationLastRow,:);
newTestFeatures = featuresNorm(validationLastRow:testLastRow,:);

% map features (in this case only add a column of ones)
newTrainingFeatures_lin = [ones(size(newTrainingFeatures(:,1))) newTrainingFeatures];
newValidationFeatures_lin = [ones(size(newValidationFeatures(:,1))) newValidationFeatures];

% F1 vs lambda for all 8 features

lambda_pol = logspace(-10, 10, 250);

newF_Score_train_lin = zeros(length(lambda_pol), 1);
newF_Score_val_lin = zeros(length(lambda_pol), 1);

for i = 1:length(lambda_pol)
    newTheta_lin = train(newTrainingFeatures_lin, trainingLabel, lambda_pol(i));
    
    p_train_lin = predict(newTheta_lin, newTrainingFeatures_lin);
    newF_Score_train_lin(i) = F_Score(trainingLabel, p_train_lin);
    
    p_val_lin = predict(newTheta_lin, newValidationFeatures_lin);
    newF_Score_val_lin(i) = F_Score(validationLabel, p_val_lin);
end

% plot F1 vs lambda for all 8 features
figure;
plot(lambda_pol, newF_Score_train_lin);
set(gca, 'XScale', 'log');
axis([10^(-10) 10^10 0 1]);
xlabel('lambda');
ylabel('F1 Score');
hold on
plot(lambda_pol, newF_Score_val_lin);
legend('Training set','Validation set');
title('All 8 features (linear)');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Non-linear Classifier with 8 features
close all
% map features (in this case only add a column of ones)
newTrainingFeatures_pol = mapFeaturesQuadratic(newTrainingFeatures, [1 2 3 4 5 6 7 8]);
newValidationFeatures_pol = mapFeaturesQuadratic(newValidationFeatures, [1 2 3 4 5 6 7 8]);

% F1 vs lambda for all 8 features

lambda_pol = logspace(-10, 10, 250); 

newF_Score_train_pol = zeros(length(lambda_pol), 1);
newF_Score_val_pol = zeros(length(lambda_pol), 1);

for i = 1:length(lambda_pol)
    newTheta_pol = train(newTrainingFeatures_pol, trainingLabel, lambda_pol(i));
    
    p_train_pol = predict(newTheta_pol, newTrainingFeatures_pol);
    newF_Score_train_pol(i) = F_Score(trainingLabel, p_train_pol);
    
    p_val_pol = predict(newTheta_pol, newValidationFeatures_pol);
    newF_Score_val_pol(i) = F_Score(validationLabel, p_val_pol);
end

% plot F1 vs lambda for all 8 features
figure;
plot(lambda_pol, newF_Score_train_pol);
set(gca, 'XScale', 'log');
axis([10^(-10) 10^10 0 1]);
xlabel('lambda');
ylabel('F1 Score');
hold on
plot(lambda_pol, newF_Score_val_pol);
legend('Training set','Validation set');
title('All 8 features (non-linear)');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Adding more training examples
close all

F_Score_train_add = zeros([length(newTestFeatures) 1]);
F_Score_val_add = zeros([length(newTestFeatures) 1]);

for i = 1:length(newTestFeatures)
    theta_more_ex = train([newTrainingFeatures ; newTestFeatures(1:i,:)], [trainingLabel ; testLabel(1:i,:)], 0);
    
    p_train_add = predict(theta_more_ex, [newTrainingFeatures ; newTestFeatures(1:i,:)]);
    F_Score_train_add(i) = F_Score([trainingLabel ; testLabel(1:i,:)], p_train_add);
    
    p_val_add = predict(theta_more_ex, newValidationFeatures);
    F_Score_val_add(i) = F_Score(validationLabel, p_val_add);
end

figure;
plot(1:length(newTestFeatures), F_Score_train_add, 1:length(newTestFeatures), F_Score_val_add);
xlabel('Number of training examples');
ylabel('F1 Score');
legend('Training set','Validation set');
title('All 8 features (non-linear)');