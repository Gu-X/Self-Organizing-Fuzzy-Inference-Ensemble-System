function [EstimatedLabels,Acc,CM]=SOFEnsemble(TrainingData,TrainingLabels,TestingData,TestingLabels,G,L,F,H)
[~,seq]=sort(rand(length(TrainingLabels),F),2,'ascend');
Input1={};
Output0=ones(size(TestingData,1),length(unique(TrainingLabels)),1);
for kk=1:1:F
    seq0=[];
    for jj=1:1:H
        seq0=[seq0;find(seq(:,jj)==kk)];
    end
    seq1{kk}=unique(seq0);
    Input1.TraData=TrainingData(seq1{kk},:);
    Input1.TraLabel=TrainingLabels(seq1{kk});
    Input1.GranLevel=G;
    Input1.ClassLabels=unique(TrainingLabels);
    Input1.BatchSize=L;
    tic
    [Output1]=SOFISplus(Input1,'Learning');
    toc
    Input2.TesData=TestingData;
    Input2.TesLabel=TestingLabels;
    Input2.BatchSize=L;
    Input2.TrainedClassifier=Output1.TrainedClassifier;
    [Output2]=SOFISplus(Input2,'Testing');
    Output0=Output0.*exp(-1*(Output2.C2PDist).^2./(2*(mean(sum(TrainingData.^2,2))-sum(mean(TrainingData,1).^2))));
end
[~,EstimatedLabels]=max(Output0,[],2);
CM=confusionmat(TestingLabels,EstimatedLabels);
Acc=sum(sum(confusionmat(TestingLabels,EstimatedLabels).*(eye(length(unique(TestingLabels))))))/length(TestingLabels);
end