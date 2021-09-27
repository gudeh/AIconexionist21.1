clearvars, close all, clc;
setdemorandstream(391418381);
%% Busca os dados de Treinamento e Teste do Cat vs Non-Cat% Lê os dados de treinamento e converte para array de números

% Lê os dados de treino 
inputTrain = h5read('train_catvnoncat.h5','/train_set_x');
imgTrain = permute(inputTrain,[3,2,1,4]);
outputTrain = h5read('train_catvnoncat.h5','/train_set_y');


% Lê os dados de teste 
inputTest = h5read('test_catvnoncat.h5','/test_set_x');
imgTest = permute(inputTest,[3,2,1,4]);
outputTest = h5read('test_catvnoncat.h5','/test_set_y');


%% Examina o Número de Classes

figure
histogram(outputTrain);
%% Mostrando Alguns Exemplos de Imagens

figure
for i = 1: 20
    subplot(4,5,i);
    A = imgTrain(:,:,:,i);
    imshow(reshape(A,[64, 64, 3]));
end
%% Separando os Dados em Trainamento e Validacao

val_split=0.1;
data = rand(1,numel(outputTrain)); 
trainIdx = randperm(numel(data), round(numel(data)*(1-val_split)));
valIdx = find(~ismember(1:numel(data), trainIdx));
X_train = imgTrain(:,:,:,trainIdx);
X_val = imgTrain(:,:,:,valIdx);
Y_train = categorical(outputTrain(trainIdx));
Y_val = categorical(outputTrain(valIdx));

X_test = imgTest;
Y_test = categorical(outputTest);

num_classes=2;
input_shape = [64,64,3];
%%

layers = [
    imageInputLayer(input_shape)
    
    convolution2dLayer(3,32)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
 
    convolution2dLayer(3,64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
 
    convolution2dLayer(3,64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.5)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MiniBatchSize',64, ...
    'MaxEpochs',150, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_val, Y_val}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'ExecutionEnvironment',"gpu", ...
    'Plots','training-progress');

%%
convnet = trainNetwork(X_train,Y_train,layers,options);
%%
Y = classify(convnet,X_train);
figure
plotconfusion(Y_train,Y);
Y = classify(convnet,X_test);
figure
plotconfusion(Y_test,Y);
%%
nTest = numel(Y_test);
figure
for i=1:nTest
    A = imgTest(:,:,:,i);
    image([0,0],[2,2],reshape(A,[64, 64, 3]));
    drawnow
    
    Prediction = classify(convnet,imgTest(:,:,:,i)); 
    if Prediction == categorical(1)
      disp('E UM GATO')
    else
      disp('NAO E UM GATO')
    end
    pause
end