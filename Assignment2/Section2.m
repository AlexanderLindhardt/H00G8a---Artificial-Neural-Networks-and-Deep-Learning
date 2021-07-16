clear all, clc, clf
train_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\lasertrain.dat');
train_mean = mean(train_set);
train_std = std(train_set);

train_set = (train_set - train_mean)/train_std;
%%
p = 50;
X_train = getTimeSeriesTrainData(train_set,p);

%%

%% Build ANN
clf, clc

algorithm = 'trainlm';
net = feedforwardnet(50, algorithm); % net with 1 hidden layers

net.trainParam.epochs = 20000;
net.divideFcn = 'divideblock';
net.trainParam.max_fail = 3;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

input = X_train(1:p-1,:)';
target = X_train(p,:)';

net = train(net, input' , target');

%% Predict original training set
clf

    y_train = sim(net, X_train(1:p-1,:))
    plot(y_train,'r')
    hold on
    plot(X_train(p,:),'b')
%% Predict future
clf
test_set = load('C:\Users\Alex_\Documents\ANN Leuven\Assignment 2\Files\laserpred.dat');

test_set = (test_set - train_mean)/train_std;

X_test = getTimeSeriesTrainData(test_set,p);
%y_test = sim(net, X_train(1:p-1, end-p+1:end));
y_test = X_train(p, end-p+1:end);
%y_test2 = sim(net, X_train(1:p-1, end-1))
y_test = [y_test sim(net,X_test(1:p-1,:))];
plot(y_test,'r')
hold on
plot(test_set,'b')
e = gsubtract(y_test(2:end), test_set(1:end-1)');
rmse = sqrt(mse(e))

clf
plot(test_set)
hold on
%predict_set = sim(net, X_train(1:p-1, end-p+1:end));
predict_set = X_train(p, end-p+1:end);

for i=1:100
    predict_set(p+i) = sim(net, predict_set(i+1:p+i-1)');
end


e = gsubtract(predict_set(p+2:end), test_set(1:end-1)');
rmse = sqrt(mse(e))
plot(predict_set(p+2:end),'r')
legend('Target','Predicted')