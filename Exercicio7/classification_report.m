function [report] = classification_report(Target,Output)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[confusionMat, order]= confusionmat(Target,Output);
report = zeros(length(order),4);
for i=1:length(order)
    support(i) = sum(confusionMat(i,:));
    tp(i) = confusionMat(i,i);
    
    precision(i) = tp(i)/sum(confusionMat(:,i));
    recall(i) = tp(i)/sum(confusionMat(i,:));
    f1_score(i) = 2*((precision(i)*recall(i))/(precision(i)+recall(i)));
        
    report(i,:) = [precision(i) recall(i) f1_score(i) support(i)];
end
    
end

