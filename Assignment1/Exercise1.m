P = [0 0 1 1; 0 1 0 1];
T = [0 0 0 1];

net = newp(P, T, 'hardlim', 'learnp');
net = init(net);


[net,tr_descr] = train(net,P,T);
net.trainParam.epochs = 20;
Pnew = [0.1;1];
sim(net,Pnew)