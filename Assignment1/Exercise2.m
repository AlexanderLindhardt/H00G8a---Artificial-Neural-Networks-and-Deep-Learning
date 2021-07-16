%% Create new dataset



clear all, clc, clf
load('Data_Problem1_regression.mat');

d1=8; d2=7; d3=7; d4=6; d5=2;
T_new = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);

data = [X1 X2 T_new];

indices = datasample(1:size(X1), 3000,'Replace',false);

train_set = data(indices(1:1000),:);
val_set = data(indices(1001:2000),:);
test_set = data(indices(2001:3000),:);
%% Plots
clf
F=scatteredInterpolant(train_set(:,1), train_set(:,2), train_set(:,3));
[x,y] = meshgrid(0:0.01:1);
%F.Method = 'nearest';
vq1 = F(x,y);
plot3(train_set(:,1),train_set(:,2),train_set(:,3),'.')
hold on
mesh(x,y,vq1)
title('Training set')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

%% Build ANN
clf, clc

algorithm = 'trainbr';
net = feedforwardnet([20 20], algorithm); % net with 2 hidden layers
%net = configure(net, train_set(:,1:2)', train_set(:,3)');
net = init(net);
net.trainParam.epochs = 800;
net.divideFcn = 'divideblock';
net.trainParam.max_fail = 10;
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio   = 0.5;
net.divideParam.testRatio  = 0;
input2 = [train_set(:,1), train_set(:,2); val_set(:,1), val_set(:,2)];
target2 = [train_set(:,3); val_set(:,3)];
net = train(net, input2' , target2');
%%
net.layers{3}
%%
clc
clf
subplot(2,1,1);
F=scatteredInterpolant(test_set(:,1), test_set(:,2), test_set(:,3));
[x1,y1] = meshgrid(0:0.01:1);
F.Method = 'nearest';
vq1 = F(x1,y1);
plot3(test_set(:,1),test_set(:,2),test_set(:,3),'mo')
hold on
mesh(x1,y1,vq1)
title('Surface of the test set')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

test_res=sim(net,test_set(:,1:2)');
subplot(2,1,2);
F=scatteredInterpolant(test_set(:,1), test_set(:,2), test_res');
[x2,y2] = meshgrid(0:0.01:1);
F.Method = 'nearest';
vq2 = F(x2,y2);
plot3(test_set(:,1),test_set(:,2),test_res','mo')
hold on
mesh(x2,y2,vq2)
title('Approximation by the network')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

%%
clf, clc
x = test_set(:,1);
y = test_set(:,2);
err = test_res' - test_set(:,3);
%err = immse(test_res', test_set(:,3));
[X,Y] = meshgrid(linspace(min(x),max(x),1000), linspace(min(y),max(y),1000));
Z = griddata(x,y,err, X, Y);
contour(X,Y,Z)

%scatter3(test_set(:,1), test_set(:,2), err)
%plot(1:1000, err)