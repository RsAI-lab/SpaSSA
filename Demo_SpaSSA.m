%==========================================================================
% G. Sun, H. Fu, et al, "SpaSSA: Superpixelwise Adaptive SSA for Unsupervised
% Spatial¨CSpectral Feature Extraction in Hyperspectral Image"

% SpaSSA on Indian Pines dataset
%==========================================================================

close all;clear all;clc;
addpath(genpath('.\Dataset'));
addpath(genpath('.\libsvm-3.18'));
addpath(genpath('.\functions'));

%% data
load('indian_pines_gt'); img_gt=indian_pines_gt;
load('Indian_pines_corrected');img=indian_pines_corrected;
[Nx,Ny,bands]=size(img);

%% superpixel segmentation
grey_img=imread('PC1_IP.tif');
grey_img=double(grey_img);
nC = 50;                       % the number of superpixels
lambda_prime = 0.5;sigma = 5.0;conn8 = 1;
[labels] = mex_ers(grey_img,nC,lambda_prime,sigma,conn8);
[height,width] = size(grey_img);

%% SpaSSA
img_SpaSSA=zeros(Nx,Ny,bands);
T1=3;      %threshold T1
T2=10;     %threshold T2
L1D=10;    %1-D embedding window of Super-1DSSA
tic;
for i=0:nC-1
    Mark=0;
    [r,c]=find(labels==i);
    index=find(labels==i);
    minr=min(r);maxr=max(r);row=length(min(r):max(r)); 
    minc=min(c);maxc=max(c);col=length(min(c):max(c));
    
    if min(row,col)<(2*T1)          %equation (17)
        L=L1D;Mark=1;               
    elseif min(row,col)>=(2*T2)
        L2=T2;
    else
        L2=floor(min(row,col)*0.5);
    end
    
    %adaptive method
    if Mark==0                      %Super-2D-SSA
        sp=zeros(Nx,Ny);
        for j=1:bands
            II=zeros(Nx,Ny);
            sup_bandimg=img(minr:maxr,minc:maxc,j);
            rec_sup_bandimg=SSA_2D(sup_bandimg,L2,L2);
            sp(minr:maxr,minc:maxc)=rec_sup_bandimg;
            II(index)=sp(index);
            img_SpaSSA(:,:,j)=img_SpaSSA(:,:,j)+II;
        end
    else                           %Super1D-SSA
        for j=1:bands
            bandimg=img(:,:,j);
            sup_bandimg=bandimg(index);
            rec_sup_bandimg=SSA(L,sup_bandimg');
            sp=img_SpaSSA(:,:,j);
            sp(index)=rec_sup_bandimg;
            img_SpaSSA(:,:,j)=sp;
        end
    end
end
toc;

%% training-test samples
Labels=img_gt(:);    
Vectors=reshape(img_SpaSSA,Nx*Ny,bands);  
class_num=max(max(img_gt))-min(min(img_gt));
trainVectors=[];trainLabels=[];train_index=[];
testVectors=[];testLabels=[];test_index=[];
rng('default');
Samp_pro=0.1;                                                         %proportion of training samples
for k=1:1:class_num
    index=find(Labels==k);                  
    perclass_num=length(index);           
    Vectors_perclass=Vectors(index,:);    
    c=randperm(perclass_num);                                      
    select_train=Vectors_perclass(c(1:ceil(perclass_num*Samp_pro)),:);    %select training samples
    train_index_k=index(c(1:ceil(perclass_num*Samp_pro)));
    train_index=[train_index;train_index_k];
    select_test=Vectors_perclass(c(ceil(perclass_num*Samp_pro)+1:perclass_num),:); %select test samples
    test_index_k=index(c(ceil(perclass_num*Samp_pro)+1:perclass_num));
    test_index=[test_index;test_index_k];
    trainVectors=[trainVectors;select_train];                    
    trainLabels=[trainLabels;repmat(k,ceil(perclass_num*Samp_pro),1)];
    testVectors=[testVectors;select_test];                      
    testLabels=[testLabels;repmat(k,perclass_num-ceil(perclass_num*Samp_pro),1)];
end
[trainVectors,M,m] = scale_func(trainVectors);
[testVectors ] = scale_func(testVectors,M,m);   

%% SVM-based classification
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
models=svmtrain(trainLabels,trainVectors,cmd);
testLabel_est= svmpredict(testLabels,testVectors, models);

%classification map
result_gt= Labels;       
for i = 1:1:length(testLabel_est)        
   result_gt(test_index(i)) = testLabel_est(i);  
end
result_map_l = reshape(result_gt,Nx,Ny);result_map=label2color(result_map_l,'india');figure,imshow(result_map);

%classification results
[OA,AA,kappa,CA]=confusion(testLabels,testLabel_est);
result=[CA*100;OA*100;AA*100;kappa*100]