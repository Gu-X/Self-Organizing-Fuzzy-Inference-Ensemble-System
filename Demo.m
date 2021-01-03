clear all
clc
close all

load exampledata.mat
TrainingData=TrainingData(1:1:6000,:);
TrainingLabels=TrainingLabels(1:1:6000);
L1=length(TestingLabels);
seq=randperm(L1);
TestingLabels=TestingLabels(seq(1:1:1000));
TestingData=TestingData(seq(1:1:1000),:);
G=9;     % the level of granularity, G;
L=2000; % the chunk Size, L;
F=5;     % the number of base learners, F;
H=2;     % the number of data pools that receive new sample per instance, H.
[EstimatedLabels,Acc,CM]=SOFEnsemble(TrainingData,TrainingLabels,TestingData,TestingLabels,G,L,F,H);
Acc      % the classification accuracy  of the classification result
CM       % the confusion matrix of the classification result
EstimatedLabels; % the estimated labels of testing data