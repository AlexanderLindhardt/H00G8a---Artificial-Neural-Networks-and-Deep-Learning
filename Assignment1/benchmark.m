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
H = 30;% Number of neurons in the hidden layer
delta_epochs = [1,14,985];% Number of epochs to train in each step
epochs = sum(delta_epochs);

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;
y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=y;% Targets. Change to yn to train on noisy data
%%
%creation of networks
for i=1:100
    
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

    net1.trainParam.epochs = epochs;
    net2.trainParam.epochs = epochs;
    net3.trainParam.epochs = epochs;
    net4.trainParam.epochs = epochs;
    net5.trainParam.epochs = epochs;
    net6.trainParam.epochs = epochs;

    tic
    net1=train(net1,x,t);
    time1(i) = toc;

    tic
    net2=train(net2,x,t);
    time2(i) = toc;

    tic
    net3=train(net3,x,t);
    time3(i) = toc;

    tic
    net4=train(net4,x,t);
    time4(i) = toc;

    tic
    net5=train(net5,x,t);
    time5(i) = toc;

    tic
    net6=train(net6,x,t);
    time6(i) = toc;

    a1 = sim(net1,x); 
    a2 = sim(net2,x); 
    a3 = sim(net3,x); 
    a4 = sim(net4,x); 
    a5 = sim(net5,x); 
    a6 = sim(net6,x); 

    err1(i) = immse(a1, t);
    err2(i) = immse(a2, t);
    err3(i) = immse(a3, t);
    err4(i) = immse(a4, t);
    err5(i) = immse(a5, t);
    err6(i) = immse(a6, t);
    
    R1(i) = regression(a1,y);
    R2(i) = regression(a2,y);
    R3(i) = regression(a3,y);
    R4(i) = regression(a4,y);
    R5(i) = regression(a5,y);
    R6(i) = regression(a6,y);

end
%%

bar([mean(R1) mean(R2) mean(R3) mean(R4) mean(R5) mean(R6)])
bar([mean(err1) mean(err2) mean(err3) mean(err4) mean(err5) mean(err6)])
bar([mean(time1) mean(time2) mean(time3) mean(time4) mean(time5) mean(time6)])