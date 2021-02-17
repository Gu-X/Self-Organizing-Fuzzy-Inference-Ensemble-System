function [Output]=SOFISplus(Input,Mode)
if strcmp(Mode,'Learning')==1
    lambda0=1;
    lambda1=2;
    [Output.TrainedClassifier]=training(Input.TraData,Input.TraLabel,Input.GranLevel,Input.BatchSize,lambda0,lambda1,Input.ClassLabels);
end
if strcmp(Mode,'Testing')==1
    Output.TrainedClassifier=Input.TrainedClassifier;
    [Output.EstLabel,Output.C2PDist]=testing(Input.TrainedClassifier,Input.TesData,Input.BatchSize);
    Output.ConfMat=confusionmat(Input.TesLabel,Output.EstLabel);
    Output.Accuracy=sum(sum(Output.ConfMat.*(eye(length(unique(Input.TesLabel))))))./length(Input.TesLabel);
end
end
function [label_est,dist]=testing(TrainedClassifier,data_test,BS)
centre=TrainedClassifier.centre;
seq=TrainedClassifier.seq;
N=length(centre);
L=size(data_test,1);
dist=zeros(L,N);
if L<=BS
    for i=1:1:N
        if isempty(centre{i})~=1
            dist(:,i)=min(pdist2(data_test,centre{i},'euclidean').^2,[],2);
        end
    end
else
    NB=floor(L/BS);
    for i=1:1:N
        for ii=1:1:NB
            dist(BS*(ii-1)+1:1:ii*BS,i)=min(pdist2(data_test(BS*(ii-1)+1:1:ii*BS,:),centre{i},'euclidean').^2,[],2);
        end
        dist(ii*BS+1:1:end,i)=min(pdist2(data_test(ii*BS+1:1:end,:),centre{i},'euclidean').^2,[],2);
    end
end
[~,label_est]=min(dist,[],2);
label_est=seq(label_est);
end
function [TrainedClassifier]=training(DTra1,LTra1,GranLevel,BatchSize,lambda0,lambda1,seq)
NB=ceil(length(LTra1)/BatchSize);
N=length(seq);
CN=zeros(N,1);
averdist=zeros(N,1);
if NB==0
    TrainedClassifier=[];
elseif NB>=2
    for kk=1:1:NB-1
        centre={};
        data_train=DTra1((kk-1)*BatchSize+1:1:(kk)*BatchSize,:);
        label_train=LTra1((kk-1)*BatchSize+1:1:(kk)*BatchSize);
        data_train1={};
        for ii=1:1:N
            centre{ii}=[];
            data_train1{ii}=data_train(label_train==seq(ii),:);
            if isempty(data_train1{ii})~=1
                [CN0,W]=size(data_train1{ii});
                dist00=pdist(data_train1{ii},'euclidean').^2;
                for tt=1:GranLevel
                    dist00(dist00>mean(dist00))=[];
                end
                if kk==1
                    averdist(ii)=mean(dist00);
                    if isnan(averdist(ii))==1
                        averdist(ii)=0;
                    end
                else
                    averdist(ii)=(CN(ii)*averdist(ii)+CN0*mean(dist00))/(CN(ii)+CN0);
                    if isnan(averdist(ii))==1
                        averdist(ii)=0;
                    end
                end
                CN(ii)=CN(ii)+CN0;
                [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
            end
        end
        if kk==1
            centre0=centre;
        else
            [centre0]=CombiningCentres(centre0,centre,averdist,N,lambda0,lambda1);
        end
    end
    data_train=DTra1((kk)*BatchSize+1:1:end,:);
    label_train=LTra1(kk*BatchSize+1:1:end);
    if length(label_train)>=BatchSize/10
        data_train1={};
        for ii=1:1:N
            data_train1{ii}=data_train(label_train==seq(ii),:);
            if isempty(data_train1{ii})~=1
                [CN0,W]=size(data_train1{ii});
                dist00=pdist(data_train1{ii},'euclidean').^2;
                for tt=1:GranLevel
                    dist00(dist00>mean(dist00))=[];
                end
                averdist(ii)=(CN(ii)*averdist(ii)+CN0*mean(dist00))/(CN(ii)+CN0);
                if isnan(averdist(ii))==1
                    averdist(ii)=0;
                end
                [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
            end
        end
        [centre0]=CombiningCentres(centre0,centre,averdist,N,lambda0,lambda1);
    end
elseif NB==1
    data_train=DTra1;
    label_train=LTra1;
    data_train1={};
    for ii=1:1:N
        data_train1{ii}=data_train(label_train==seq(ii),:);
        if isempty(data_train1{ii})~=1
            dist00=pdist(data_train1{ii},'euclidean').^2;
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];
            end
            averdist(ii)=mean(dist00);
            if isnan(averdist(ii))==1
                averdist(ii)=0;
            end
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
        end
    end
    centre0=centre;
end
TrainedClassifier.seq=seq;
TrainedClassifier.centre=centre0;
TrainedClassifier.averdist=averdist;
TrainedClassifier.NoC=N;
end
function [centre0]=CombiningCentres(centre0,centre,thresholddistance,N,lambda0,lambda1)
La1=[];
La2=[];
CC1=[];
CC2=[];
for ii=1:1:N
    CC1=[CC1;centre0{ii}];
    CC2=[CC2;centre{ii}];
    La1=[La1;ones(size(centre0{ii},1),1)*ii];
    La2=[La2;ones(size(centre{ii},1),1)*ii];
end
dist11=pdist2(CC1,CC2).^2;
for ii=1:1:N
    seq11=find(La1==ii);
    seq22=find(La1~=ii);
    seq33=find(La2==ii);
    if isempty(seq11)~=1 && isempty(seq11)~=1
%%
        dist1=dist11(seq11,seq33);
        seq1=min(dist1,[],1);
        seq2=find(seq1>=thresholddistance(ii)*lambda0);
        dist2=dist11(seq22,seq33);
        dist3=repmat(thresholddistance(La1(seq22))*lambda1,1,length(seq33));
        dist4=dist2-dist3;
        seq44=min(dist4,[],1);
        seq4=find(seq44<=0);
        centre0{ii}=[centre0{ii};centre{ii}(unique([seq2,seq4]),:)];
%%
%    dist1=dist11(seq11,seq33);
%         seq1=min(dist1,[],1);
%         seq2=find(seq1>=thresholddistance(ii)*lambda0);
%         dist2=dist11(seq22,seq33);
%         dist3=repmat(thresholddistance(La1(seq22))*lambda1,1,length(seq33));
%         dist4=dist2-dist3;
%         seq44=min(dist4,[],1);
%         seq4=find(seq44<=0);
%         dist3=dist11(seq11,seq33(seq4));
%         seq6=min(dist3,[],1);
%         seq5=find(seq6<=thresholddistance(ii)*0.1);
%         seq4(seq5)=[];
%         centre0{ii}=[centre0{ii};centre{ii}(unique([seq2,seq4]),:)];
    else
        centre0{ii}=[centre0{ii};centre{ii}];
    end
end
end
function [centre]=online_training_Euclidean(data,averdist)
[L,W]=size(data);
centre=data(1,:);
member=1;
for ii=2:1:L
    [dist3,pos3]=min(pdist2(data(ii,:),centre,'euclidean').^2);
    if dist3>averdist
        centre(end+1,:)=data(ii,:);
        member(end+1,1)=1;
    else
        centre(pos3,:)=(member(pos3,1)*centre(pos3,:)+data(ii,:))/(member(pos3,1)+1);
        member(pos3,1)=member(pos3,1)+1;
    end
end
end