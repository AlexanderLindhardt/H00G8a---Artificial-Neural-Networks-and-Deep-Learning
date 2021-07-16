clear all, clc, clf
train_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\lasertrain.dat');
test_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\laserpred.dat');
hold on
plot(train_set,'b')
plot(1001:1100, test_set,'r')
axis([0 1110 0 300])
%%
train_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\lasertrain.dat');
train_mean = mean(train_set);
train_std = std(train_set);
train_set = (train_set - train_mean)/train_std;

p = 50;
X_train = getTimeSeriesTrainData(train_set, p);

input = X_train(1:p-1,:);
target = X_train(p,:);


numFeatures = p-1;
numResponses = 1;
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(input,target,layers,options);
%%
clc
clf
test_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\laserpred.dat');
test_set = (test_set - train_mean)/train_std;
X_test = getTimeSeriesTrainData(test_set,p);


plot(test_set)
hold on
net = predictAndUpdateState(net,X_train);
predict_set = X_train(p, end-p+1:end);

for i=1:100
    %[net, YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    [net, predict_set(p+i)] = predictAndUpdateState(net, predict_set(i+1:p+i-1)','ExecutionEnvironment','cpu');
end

e = gsubtract(predict_set(p+2:end), test_set(1:end-1)');
rmse = sqrt(mse(e))
plot(predict_set(p+1:end),'r')
legend('Target','Predicted')
%%
net = predictAndUpdateState(net, X_train);
[net,YPred] = predictAndUpdateState(net,ytrain(end));

numTimeStepsTest = numel(Xtest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = train_std*YPred + train_mean;
YTest = test_set(2:end);
rmse = sqrt(mean((YPred-YTest).^2))