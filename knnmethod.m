clear all; close all; clc

A = csvread('train.csv');

B = A(1:33600,2:785)'; %Train
C = A(33601:end,2:785)';%Test 42000


trainImages = B;
trainLabels = A(1:33600,1);

N = 784;  
K = 100; % can be any other value 
testImages = C;
testLabels = A(33601:end,1);

trainLength = length(trainImages);  
testLength = size(testImages,2);  
testResults = linspace(0,0,length(testImages));  
compLabel = linspace(0,0,K);  
tic;  

for i=1:testLength  
    curImage = repmat(testImages(:,i),1,trainLength);  
    curImage = abs(trainImages-curImage);  
    comp=sum(curImage);  
    [sortedComp,ind] = sort(comp);  
    for j = 1:K  
        compLabel(j) = trainLabels(ind(j));  
    end  
    table = tabulate(compLabel);  
    [maxCount,idx] = max(table(:,2));  
    testResults(i) = table(idx);    
  
    disp(testResults(i));  
    disp(testLabels(i));  
end  

% Compute the error on the test set  
error=0;  
for i=1:testLength  
  if (testResults(i) ~= testLabels(i))  
    error=error+1;  
  end  
end  
  
%Print out the classification error on the test set  
error/testLength  
toc;  
disp(toc-tic); 