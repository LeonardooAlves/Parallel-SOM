%Limpeza de tela
 clear variables;
 clc;

%% ---------------------------- Parâmetros Iniciais -----------------------
%% Algorithm

%Defining data points (input space)
SourceNodes = 2000; %number os input data points
x1 = rand(SourceNodes,1); %data point dimension 1
x2 = rand(SourceNodes,1); %data point dimension 2
x3 = rand(SourceNodes,1); %data point dimension 1
x4 = rand(SourceNodes,1); %data point dimension 2
% x5 = rand(SourceNodes,1); %data point dimension 1
% x6 = rand(SourceNodes,1); %data point dimension 2
% x7 = rand(SourceNodes,1); %data point dimension 1
% x8 = rand(SourceNodes,1); %data point dimension 2
% x9 = rand(SourceNodes,1); %data point dimension 1
% x10 = rand(SourceNodes,1); %data point dimension 2
% x11 = rand(SourceNodes,1); %data point dimension 1
% x12 = rand(SourceNodes,1); %data point dimension 2
% x13 = rand(SourceNodes,1); %data point dimension 1
% x14 = rand(SourceNodes,1); %data point dimension 2

% x1 = sort(x1);
% x2 = sort(x2);

% values = struct('values', x1);
% In1 = struct('time', [], 'signals',values);
% values = struct('values', x2);
% In2 = struct('time', [], 'signals',values);
% values = struct('values', x3);
% In3 = struct('time', [], 'signals',values);
% values = struct('values', x4);
% In4 = struct('time', [], 'signals',values);
% values = struct('values', x5);
% In5 = struct('time', [], 'signals',values);
% values = struct('values', x6);
% In6 = struct('time', [], 'signals',values);
% values = struct('values', x7);
% In7 = struct('time', [], 'signals',values);
% values = struct('values', x8);
% In8 = struct('time', [], 'signals',values);
% values = struct('values', x9);
% In9 = struct('time', [], 'signals',values);
% values = struct('values', x10);
% In10 = struct('time', [], 'signals',values);
% values = struct('values', x11);
% In11 = struct('time', [], 'signals',values);
% values = struct('values', x12);
% In12 = struct('time', [], 'signals',values);
% values = struct('values', x13);
% In13 = struct('time', [], 'signals',values);
% values = struct('values', x14);
% In14 = struct('time', [], 'signals',values);

%Defining neurons
Neurons = 25; %number of neurons
w1 = rand(1,Neurons); %initial weight dimension 1
w2 = rand(1,Neurons); %initial weight dimension 2
w3 = rand(1,Neurons); %initial weight dimension 1
w4 = rand(1,Neurons); %initial weight dimension 2
% w5 = rand(1,Neurons); %initial weight dimension 1
% w6 = rand(1,Neurons); %initial weight dimension 2
% w7 = rand(1,Neurons); %initial weight dimension 1
% w8 = rand(1,Neurons); %initial weight dimension 2
% w9 = rand(1,Neurons); %initial weight dimension 1
% w10 = rand(1,Neurons); %initial weight dimension 2
% w11 = rand(1,Neurons); %initial weight dimension 1
% w12 = rand(1,Neurons); %initial weight dimension 2
% w13 = rand(1,Neurons); %initial weight dimension 1
% w14 = rand(1,Neurons); %initial weight dimension 2


w1init = w1;
w2init = w2;
% Neuron coordinators in the 2D lattice
[rows,columns] = meshgrid(1:sqrt(Neurons));
wx = zeros(1,Neurons);
wy = zeros(1,Neurons);
wx(1,1:Neurons) = rows(1:Neurons);    
wy(1,1:Neurons) = columns(1:Neurons);

%Iterations and rates
IterationsOrdering = 4000; %iterations on ordering phase
IterationsCovergence = (Neurons*500)-1; %iterations on convergence phase
TotalIterations = IterationsOrdering+IterationsCovergence;
Dim = sqrt(Neurons); %SOM Dimension limit 
S0 = Dim; %to allow the neighborhood init with the entire map/initial neighborhood size in the radious
T1 = IterationsOrdering/log(S0); %neighborhood size time
N0 = 0.1; %initial learning rate
N_limit = 0.01; %learning rate minimum value
S_limit = 1; %neighborhood size minimum value
T2 = IterationsOrdering/log(N0/N_limit); %learning rate time


%% Harware
ts = 1; %sampling rate

%Competitive Process
%Data bits
nb = 16; %number of bits | Manhattan distance
bp = 12; %fractional part | Manhattan distance
%Coords bits
bpCoord = 0; %fractional part of neuron coordinates
nbCoord = ceil(log2(Dim))+1 +bpCoord; %number of bits of neuron coordinates. Its multiplied by 2 because of the multiplier in the distance measurement
nbCoordMult = ceil(log2((abs(max(wx) - min(wx))+abs(max(wy) - min(wy)))^2));%ceil(log2(((max(wx) - min(wx))^2+(max(wy) - min(wy))^2)));
bpCoordMult = 0;
%Memory (Neighbour) Bits
bpSNMemory = bp;
nBits = 9; %number of bits for the LUTs used for neighbor size (S) and learning rate (N)



nbIteration = ceil(log2(IterationsOrdering));
bpIteration = 0;
nbMuxTree = Dim;
%Cooperating Process
bpMult = bp;
nbMult = bpMult+2;


%Delays
dSub_CompetitiveProcess = 0;
dMult_CompetitiveProcess = 0;
dAdd_CompetitiveProcess = 0;
dComparator_CompetitiveProcess = 0;
dMUX_CompetitiveProcess = 0;
dLogical_CompetitiveProcess = 0;


dMemoryNeighbor = 1;
dMemoryLearning = 1;
dMemoryNeighborFunc = 1;
dAnd_CooperatingProcess = 0;
dOR_CooperatingProcess = 0;
dSub_CooperatingProcess = 0;
dMult_CooperatingProcess = 0;
dMultCoord_CooperatingProcess = 0;
dMultAddress_CooperatingProcess = 0;
dAdd_CooperatingProcess = 0;
dComparator_CooperatingProcess = 0;
dMUX_CooperatingProcess = 0;
dCast_CooperatingProcess = 0;
dDiv_CooperatingProcess = 0;
dInverter_CooperatingProcess = 0;
dGain_CooperatingProcess = 0;
dDelay_CooperatingProcess = 1;
dDelayPipe_CooperatingProcess = 1;

d3=0; d4=0; 


%Memories
nBitsNSmemories = (ceil(log2(TotalIterations)));
n = 0:1:(2^nBitsNSmemories)-1;
% Update the size of neighborhood.
S_init = S0*exp(-1 * ( n / T1 ) );
for i=1:(2^nBitsNSmemories)
    if S_init(1, i) < S_limit
          S_init(1, i) = S_limit;
    end
end
S = 1 ./ (2 .*(S_init .* S_init));
% Update the learning-rate.
N = N0*exp(-1 * ( n / T2 ) );
for i=1:(2^nBitsNSmemories)
    if N(1, i) < N_limit
          N(1, i) = N_limit;
    end
end
% MaxCoordDistance = (max(wx) - min(wx))^2;
% domain = linspace(0, 2^ceil(log2(MaxCoordDistance)), 2^nBits);
% h = exp(-1*domain); %Neighborhood function
MaxCoordDistance = (abs(max(wx) - min(wx))+abs(max(wy) - min(wy)))^2;%((max(wx) - min(wx))^2+(max(wy) - min(wy))^2);
domain = linspace(0, MaxCoordDistance, 2^nBits);
h = exp(-1*domain); %Neighborhood function

nbGainBlock = bpSNMemory+ceil(log2((2^nBits)/MaxCoordDistance))+1;

%% Plots Iniciais
figure;
    
% Plot graph with initial weights.
subplot(2,2,2); 
neighborhood(w1,w2,Dim,Dim);
xlabel('w1');
ylabel('w2');
title('Weights in Initial State','FontSize',14);
axis([0,1,0,1]);
    
% Plot graph with input space.
subplot(2,2,1); 
plot(x1(:,1),x2(:,1),'*');
xlabel('x1');
ylabel('x2');
title('Input Space','FontSize',14);
axis([0,1,0,1]);
    
% Summary of parameters.
subplot(2,2,3);
plot(0,0,'w.');
s = sprintf('S0 = %.2f\n\n',S0);
s = sprintf('%sT1 = %.2f\n\n',s,T1);
s = sprintf('%sN0 = %.2f\n\n',s,N0);
s = sprintf('%sT2 = %.2f\n\n',s,T2);
s = sprintf('%sNeurons = %d\n\n',s,Neurons);
s = sprintf('%sSource Nodes = %d\n\n',s,SourceNodes);
s = sprintf('%sIterations of Ordering Phase = %d',s,IterationsOrdering);
text(0.1,0.5,s);
axis([0,1,0,1]);
title('Parameters','FontSize',14);
drawnow;
    
%Implementação do livro
% History of size of neighborhood and learning-rate.
S_hist = zeros(1,IterationsOrdering+(Neurons*500));
N_hist = zeros(1,IterationsOrdering+(Neurons*500));
    
% History of weight vectors.
w1_hist = zeros(IterationsOrdering+(Neurons*500),Neurons);
w2_hist = zeros(IterationsOrdering+(Neurons*500),Neurons);
w1_hist(1,:) = w1init;
w2_hist(1,:) = w2init;
    
% History of winning neurons.
win_hist = zeros(1,IterationsOrdering+(Neurons*500));
win_bmu = zeros(2,IterationsOrdering+(Neurons*500));
win_dist = zeros(1,IterationsOrdering+(Neurons*500));
    
%% Ordering Phase.
    ex = 0;
    for g=0:(IterationsOrdering-1)
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Search minor euclidean distance.
%         dist = sqrt(((x1(ex,1)-w1init).^2) + ((x2(ex,1)-w2init).^2));
        dist = abs(x1(ex,1)-w1init) + abs(x2(ex,1)-w2init);
        [winDist,winning] = min(dist);
        [oout,indice] = sort(dist);
        
        % Cooperative Process and Updating.
        
        % Update the size of neighborhood.
        Ss = S0*exp(-1 * ( g / T1 ) );
        % Update the learning-rate.
        Nn = N0*exp(-1 * ( g / T2 ) );
        % Inferior limit of learning-rate.
        if Nn < N_limit
            Nn = N_limit;
        end
        % Inferior limit of size of neighborhood.
        if Ss < S_limit
            Ss = S_limit;
        end
        % Euclidean distance with winning neuron to square.
        distp2 = (( wx - wx(1,winning) ).^2) + (( wy - wy(1,winning) ).^2);
        % Difference with input.
        dif1 = x1(ex,1) - w1init; 
        dif2 = x2(ex,1) - w2init;
        % Neighborhood Function.
        H = exp(-1 * ( distp2 ./ (2*Ss*Ss) ) );
        % Update weights.
        w1init = w1init + Nn*H.*dif1;
        w2init = w2init + Nn*H.*dif2;  
        % Save history of size of neighborhood and learning-rate.
        S_hist(1,g+1) = Ss;
        N_hist(1,g+1) = Nn;
        % Save history of weight vectors.
        w1_hist(g+2,:) = w1init;
        w2_hist(g+2,:) = w2init;
        % Save history of winning neurons.
        win_hist(1,g+1) = winning;
        win_bmu(1,g+1) = indice(1);
        win_bmu(2,g+1) = indice(2);
        win_dist(1,g+1) = winDist;
    end
    
    figure;
    subplot(2,2,1)
    neighborhood(w1init,w2init,Dim,Dim);
    
%% Convergence Phase.
    % Maintain learning-rate in a positive small value.
    Nn = N_limit;
    % Maintain neighborhood containing nearest neighbors only.
    Ss = S_limit;
    
    ex = 0;
    for g=(IterationsOrdering):((Neurons*500)-1+IterationsOrdering)
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Search minor euclidean distance.
%         dist = sqrt(((x1(ex,1)-w1init).^2) + ((x2(ex,1)-w2init).^2));
        dist = abs(x1(ex,1)-w1init) + abs(x2(ex,1)-w2init);
        [winDist,winning] = min(dist);
        [oout,indice] = sort(dist);
        
        % Cooperative Process and Updating.
        
        % Euclidean distance with winning neuron, to square.
        distp2 = (( wx - wx(1,winning) ).^2) + (( wy - wy(1,winning) ).^2);
        % Difference with input.
        dif1 = x1(ex,1) - w1init; 
        dif2 = x2(ex,1) - w2init;
        % Neighborhood Function.
        H = exp(-1 * ( distp2 ./ (2*Ss*Ss) ) );
        % Update weights.
        w1init = w1init + Nn*H.*dif1;
        w2init = w2init + Nn*H.*dif2;  
        % Save history of size of neighborhood and learning-rate.
        S_hist(1,g+1) = Ss;
        N_hist(1,g+1) = Nn;
        % Save history of weight vectors.
        w1_hist(g+2,:) = w1init;
        w2_hist(g+2,:) = w2init;
        % Save history of winning neurons.
        win_hist(1,g+1) = winning;        
        win_bmu(1,g+1) = indice(1);
        win_bmu(2,g+1) = indice(2);
        win_dist(1,g+1) = winDist;
    end
    

% Plot graph with weights in the end of convergence phase.
subplot(2,2,3)
neighborhood(w1init,w2init,Dim,Dim);
    
% Plot graph with history of learning-rate.
subplot(2,2,2);
plot(0:1:IterationsOrdering+(Neurons*500)-1,N_hist);
ylabel('N');
xlabel('n');
grid;
title('Learning-rate','FontSize',14);
axis([0,IterationsOrdering+(Neurons*500)-1,0,N0]);
    
% Plot graph with history of size of neighborhood.
subplot(2,2,4);
plot(0:1:IterationsOrdering+(Neurons*500)-1,S_hist);
ylabel('S');
xlabel('n');
grid;
title('Size of Neighborhood','FontSize',14);
drawnow;

