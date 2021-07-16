clear all
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'trainbfg'
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

% Configuration:
alg1 = 'traingd';% First training algorithm to use
alg2 = 'traingda';% Second training algorithm to use
alg3 = 'traincgf';
alg4 = 'traincgp';
alg5 = 'trainbfg';
alg6 = 'trainlm';
H = 50;% Number of neurons in the hidden layer
delta_epochs = [1,14,985];% Number of epochs to train in each step
epochs = sum(delta_epochs);

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;
y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=y;% Targets. Change to yn to train on noisy data

%creation of networks
net1=feedforwardnet(H,alg1);% Define the feedfoward net (hidden layers)
net2=feedforwardnet(H,alg2);
net3=feedforwardnet(H,alg3);
net4=feedforwardnet(H,alg4);
net5=feedforwardnet(H,alg5);
net6=feedforwardnet(H,alg6);

net1=configure(net1,x,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);
net3=configure(net3,x,t);
net4=configure(net4,x,t);
net5=configure(net5,x,t);
net6=configure(net6,x,t);

net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net2.divideFcn = 'dividetrain';
net3.divideFcn = 'dividetrain';
net4.divideFcn = 'dividetrain';
net5.divideFcn = 'dividetrain';
net6.divideFcn = 'dividetrain';

net1=init(net1);% Initialize the weights (randomly)

net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

net3.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};

net4.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};

net5.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net5.lw{2,1}=net1.lw{2,1};
net5.b{1}=net1.b{1};
net5.b{2}=net1.b{2};

net6.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net6.lw{2,1}=net1.lw{2,1};
net6.b{1}=net1.b{1};
net6.b{2}=net1.b{2};
%% Training and simulation ALL
net1.trainParam.epochs = epochs;
net2.trainParam.epochs = epochs;
net3.trainParam.epochs = epochs;
net4.trainParam.epochs = epochs;
net5.trainParam.epochs = epochs;
net6.trainParam.epochs = epochs;

tic
net1=train(net1,x,t);
time1 = toc;

tic
net2=train(net2,x,t);
time2 = toc;

tic
net3=train(net3,x,t);
time3 = toc;

tic
net4=train(net4,x,t);
time4 = toc;

tic
net5=train(net5,x,t);
time5 = toc;

tic
net6=train(net6,x,t);
time6 = toc;

a1 = sim(net1,x); 
a2 = sim(net2,x); 
a3 = sim(net3,x); 
a4 = sim(net4,x); 
a5 = sim(net5,x); 
a6 = sim(net6,x); 
%%
err1 = immse(a1, t);
err2 = immse(a2, t);
err3 = immse(a3, t);
err4 = immse(a4, t);
err5 = immse(a5, t);
err6 = immse(a6, t);
R1 = regression(a3,y)

%% Plot ALL outputs and compare with function
figure
subplot(3,2,1);
plot(x,y,'bx',x,a1,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg1]);
axis([0 10 -1.5 1.5])
legend('target',alg1,'Location','north');

subplot(3,2,2);
plot(x,y,'bx',x,a2,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg2]);
axis([0 10 -1.5 1.5])
legend('target',alg2,'Location','north');

subplot(3,2,3);
plot(x,y,'bx',x,a3,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg3]);
axis([0 10 -1.5 1.5])
legend('target',alg3,'Location','north');

subplot(3,2,4);
plot(x,y,'bx',x,a4,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg4]);
axis([0 10 -1.5 1.5])
legend('target',alg4,'Location','north');

subplot(3,2,5);
plot(x,y,'bx',x,a5,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg5]);
axis([0 10 -1.5 1.5])
legend('target',alg5,'Location','north');

subplot(3,2,6);
plot(x,y,'bx',x,a6,'r'); % plot the sine function and the output of the network
title([num2str(epochs),' epochs',' with ', alg6]);
axis([0 10 -1.5 1.5])
legend('target',alg6,'Location','north');

%% Plot ALL linear regression
figure
subplot(2,3,1);
postregm(a1,y,alg1);

subplot(2,3,2);
postregm(a2,y,alg2);

subplot(2,3,3);
postregm(a3,y,alg3);

subplot(2,3,4);
postregm(a4,y,alg4);

subplot(2,3,5);
postregm(a5,y,alg5);

subplot(2,3,6);
postregm(a6,y,alg6);

%% training and simulation
net1.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
net2.trainParam.epochs=delta_epochs(1);
net1=train(net1,x,y);   % train the networks
net2=train(net2,x,t);
a11=sim(net1,x); a21=sim(net2,x);  % simulate the networks with the input vector x

net1.trainParam.epochs=delta_epochs(2);
net2.trainParam.epochs=delta_epochs(2);
net1=train(net1,x,y);
net2=train(net2,x,t);
a12=sim(net1,x); a22=sim(net2,x);

net1.trainParam.epochs=delta_epochs(3);
net2.trainParam.epochs=delta_epochs(3);
net1=train(net1,x,y);
net2=train(net2,x,t);
a13=sim(net1,x); a23=sim(net2,x);
%% Plots
subplot(3,3,1);
plot(x,t,'bx',x,a11,'r',x,a21,'g'); % plot the sine function and the output of the networks
title([num2str(epochs(1)),' epochs']);
legend('target',alg1,alg2,'Location','north');
subplot(3,3,2);
postregm(a11,y); % perform a linear regression analysis and plot the result
subplot(3,3,3);
postregm(a21,y);

subplot(3,3,4);
plot(x,t,'bx',x,a12,'r',x,a22,'g');
title([num2str(epochs(2)),' epoch']);
legend('target',alg1,alg2,'Location','north');
subplot(3,3,5);
postregm(a12,y);
subplot(3,3,6);
postregm(a22,y);
%
subplot(3,3,7);
plot(x,t,'bx',x,a13,'r',x,a23,'g');
title([num2str(epochs(3)),' epoch']);
legend('target',alg1,alg2,'Location','north');
subplot(3,3,8);
postregm(a13,y);
subplot(3,3,9);
postregm(a23,y);
