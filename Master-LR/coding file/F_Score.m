function F1 = F_Score(known, predicted)

confMat = confusionmat(known, predicted);
truePos = confMat(2,2);
falsePos = confMat(1,2);
falseNeg = confMat(2,1);
precision = truePos/(truePos+falsePos);
recall = truePos/(truePos+falseNeg);
F1 = (2*precision*recall)/(precision+recall);
end