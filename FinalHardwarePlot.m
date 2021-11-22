figure;
% Plot graph with weights after ordering phase.
subplot(2,2,1)
neighborhood(w1_histH(IterationsOrdering,:),w2_histH(IterationsOrdering,:),Dim,Dim);
xlabel('w1');
ylabel('w2');
title('End of Ordering Phase','FontSize',14);
axis([0,1,0,1]);
    
% Plot graph with weights after convergence phase.
subplot(2,2,3)
plot(0,0,'w.');
neighborhood(w1_histH(end,:),w2_histH(end,:),Dim,Dim);
xlabel('w1');
ylabel('w2');
title('End of Convergence Phase','FontSize',14);
axis([0,1,0,1]);
    
% Plot graph with history of learning-rate.
subplot(2,2,2);
plot(0:1:IterationsOrdering+(Neurons*500)-1,N_histH);
ylabel('N');
xlabel('n');
grid;
title('Learning-rate','FontSize',14);
axis([0,IterationsOrdering+(Neurons*500)-1,0,N0]);
    
% Plot graph with history of size of neighborhood.
subplot(2,2,4);
plot(0:1:IterationsOrdering+(Neurons*500)-1,S_histH);
ylabel('S');
xlabel('n');
grid;
title('Size of Neighborhood','FontSize',14);
axis([0,IterationsOrdering+(Neurons*500)-1,0,S0]);
drawnow;

w1errorDist = abs(w1_histH(end,:) - w1init).^2;
w2errorDist = abs(w2_histH(end,:) - w2init).^2;
w1error = w1errorDist./Neurons;
w2error = w2errorDist./Neurons;
error = [w1error; w2error];

% %Para validar com matlab
% matalabWeights = net.IW;
% hardwareWeights = [flipud(w1_histH(end,:)') flipud(w2_histH(end,:)') flipud(w3_histH(end,:)') flipud(w4_histH(end,:)')];
% D = abs(matalabWeights{1,1}-hardwareWeights).^2;
% MSE = sum(D(:))./numel(matalabWeights);