%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%
clf
clc
T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
%%
n=21;
converge_time = 0;
x = linspace(-1,1,n);
y = linspace(-1,1,n);
[X, Y] = meshgrid(x,y);
c=cat(2,X',Y');
Xi=reshape(c,[],2)';
for i=1:n*n
    [y,Pf,Af] = sim(net,{1 100},{},Xi(:,i));   % simulation of the network for 50 timesteps              
    record=[Xi(:,i) cell2mat(y)];   % formatting results  
    for j=1:size(record, 2)
        i
        if (ismember(record(:,j)', T', 'rows')) || isequal(record(:,j), [-1 1;-1 0;0 -1; 0 0;0 1;1 0]')
            converge_time = converge_time + j;
            break
        end
    end
    start=Xi(:,i);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,end),record(2,end),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
Average_convergence_iterations = converge_time/(n*n)
%%
%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
clf
clc
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n=5;
x = linspace(-1,1,n);
y = linspace(-1,1,n);
z = linspace(-1,1,n);
[X, Y, Z] = ndgrid(x, y, z);
Xi = [X(:), Y(:), Z(:)]';
converge_time = 0;
for i=1:n*n*n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 300},{},Xi(:,i));       % simulation of the network  for 50 timesteps
    record=[Xi(:,i) cell2mat(y)];       % formatting results
        for j=1:size(record, 2)
        if (ismember(record(:,j)', T', 'rows'))
            converge_time = converge_time + j;
            break
        end
    end
    start=Xi(:,i);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,end),record(2,end),record(3,end),'go');  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northwest');
title('Time evolution in the phase space of 3d Hopfield model');
Average_convergence_iterations = converge_time/(n*n*n)


%% Handwritten digits
noise = 2;
numiter = 10;
hopdigit_v2(noise,numiter)