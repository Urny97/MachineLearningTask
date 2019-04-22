function F_Score = F_Score(iets1, iets2)

[confMat,order] = confusionmat(iets1,iets2);

% Recall
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
end
recall = sum(recall)/size(confMat,1);

% Precision
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
precision = sum(precision)/size(confMat,1);

F_Score = (2*precision*recall)/(precision+recall);