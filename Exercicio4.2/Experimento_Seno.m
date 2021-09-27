%% Brincadeiras e Experiências com o Seno (e Cosseno)
%%
clearvars; close all;
%% Gera os dados de entrada
% 100 ângulos aleatórios entre 0 e 2pi

x = rand(1,100)*2*pi;
%% Gera os dados de saída
% 100 senos com ruído gaussiano média 0 e desvio padrao de 0.1

y = sin(x)+randn(1,100)*0.1;
%% Plota o conjunto de treinamento

figure
scatter(x,y,20,'red','filled')
xlabel('Angulos')
ylabel('Seno')
hold on
%% Cria dados para teste da rede 
% 361 ângulos entre 0 e 360

X = linspace(0,2*pi,361);
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% Todos os dados para treinamento
% 
% Taxa de aprendizado = 0.1
% 
% Treinamento Gradiente Descendente

net = newff(x,y,[10],{'tansig','purelin'},'traingd');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1;
net.divideFcn='';
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T);
%% Cria a Rede Neural
% 6 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% Todos os dados para treinamento
% 
% Taxa de aprendizado = 0.1
% 
% Treinamento Gradiente Descendente
%%
net = newff(x,y,[6],{'tansig','purelin'},'traingd');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1;
net.divideFcn='';
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'k');
%% Cria a Rede Neural
% 100 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% Todos os dados para treinamento
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente
%%
net = newff(x,y,[100],{'tansig','purelin'},'traingd');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01; % É necessário reduzir a taxa de aprendizado? Porque?
net.divideFcn='';
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'r');
%% Cria a Rede Neural
% 5 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% Todos os dados para treinamento
% 
% Taxa de aprendizado = 0.1
% 
% Taxa de momento = 0.9
% 
% Treinamento Gradiente Descendente com Momento
%%
net = newff(x,y,[5],{'tansig','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1;
net.trainParam.mc = 0.9;
net.divideFcn='';
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'b');
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento

net = newff(x,y,[10],{'tansig','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.trainParam.max_fail = 50;
hold off
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'g');
%% Gera os dados de entrada
% 10000 ângulos aleatórios entre 0 e 2pi
%%
x = rand(1,10000)*2*pi;
%% Gera os dados de saída
% 10000 senos com ruído gaussiano média 0 e desvio padrao de 0.1

y = sin(x)+randn(1,10000)*0.1;
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento

net = newff(x,y,[10],{'tansig','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.trainParam.max_fail = 6;
hold off
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'g');
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento
% 
% Inicializa o Pesos com ZERO
%%
net = newff(x,y,[10],{'tansig','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.divideFcn = '';


net.IW
net.IW{1,1}
net.IW{1,1}=zeros(10,1);
net.IW{1,1}
net.LW
net.LW{2,1}
net.LW{2,1}=zeros(1,10);
net.b
net.b{1,1}
net.b{1,1}=zeros(10,1);
net.b{2,1}=[0];
%% Treina a Rede Neural

net=train(net,x,y);
%% Gera os dados de entrada
% 100 ângulos aleatórios entre 0 e 2pi
%%
x = rand(1,100)*2*pi;
%% Gera os dados de saída
% 100 senos com ruído gaussiano média 0 e desvio padrao de 0.1

y = sin(x)+randn(1,100)*0.1;
%% Plota o conjunto de treinamento

figure
plot(x,y,'*')
xlabel('Angulos')
ylabel('Seno')
hold on
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento

net = newff(x,y,[10],{'logsig','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.trainParam.max_fail = 60;
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'r');
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento
%%
net = newff(x,y,[100],{'purelin','purelin'},'traingdm');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.divideFcn=''
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T,'r');
%% Cria a Rede Neural
% 10 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% 70% dos dados para treinamento
% 
% 15% dos dados para validação
% 
% 15% dos dados para teste
% 
% Taxa de aprendizado = 0.01
% 
% Treinamento Gradiente Descendente com Momento
%%
net = newff(x,y,[6],{'logsig','purelin'},'traingdx');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.9;
net.trainParam.max_fail = 30;
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% 
%%
X = linspace(-pi,3*pi,720);
T = net(X);
%% Plota os resultados de teste

plot(X,T);
%%
XTrain=reshape(x,[1,1,1,100]);
YTrain=y';

xval=linspace(0,2*pi,50);
yval=sin(xval)+randn(1,50)*0.1;
XValidation=reshape(xval,[1,1,1,50]);
YValidation=yval';

XTest=reshape(X,[1,1,1,720]);

layers = [
imageInputLayer([1 1 1])
fullyConnectedLayer(100)
reluLayer
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(1)
regressionLayer];

miniBatchSize  = 5;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
'MaxEpochs',3000, ...
'MiniBatchSize',miniBatchSize, ...
'InitialLearnRate',1e-2, ...
'Momentum',0.8, ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.5, ...
'LearnRateDropPeriod',1000, ...
'Shuffle','every-epoch', ...
'ValidationData',{XValidation,YValidation}, ...
'ValidationFrequency',validationFrequency, ...
'ValidationPatience',100,...
'Plots','training-progress', ...
'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

YPredicted = predict(net,XTest);
ytest=YPredicted';
figure
plot(x,y,'o');
hold on
plot(X,ytest);
%% Gera os dados de saída
% 100 senos com ruído gaussiano média 0 e desvio padrao de 0.1
%%
y = [sin(x)+randn(1,100)*0.1; cos(x)+randn(1,100)*0.1];
%% Plota o conjunto de treinamento

figure
plot(x,y(1,:),'*',x,y(2,:),'o')
xlabel('Angulos')
ylabel('Seno/Cosseno')
legend('seno','cosseno')
hold on
%% Cria dados para teste da rede 
% 361 ângulos entre 0 e 360

X = linspace(0,2*pi,361);
%% Cria a Rede Neural
% 6 neurônios na camada intermediária
% 
% tangente hiperbólica e linear
% 
% Todos os dados para treinamento
% 
% Taxa de aprendizado = 0.1
% 
% Treinamento Gradiente Descendente

net = newff(x,y,[6],{'tansig','purelin'},'traingdx');
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1;
net.divideFcn='';
%% Treina a Rede Neural

net=train(net,x,y);
%% Simula a Rede Treinada

T = net(X);
%% Plota os resultados de teste

plot(X,T);