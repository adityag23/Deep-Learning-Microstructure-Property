function [net,info]=create_train_networks(Xtrain,Xval,net_vec)
%% Create 3 layer network
if size(net_vec,2)==3
layers = [
    imageInputLayer([1,231],"Name","featureinput")
    fullyConnectedLayer(net_vec(1),"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(net_vec(2),"Name","fc_21")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(net_vec(3),"Name","fc_22")
    reluLayer("Name","relu_22")
    fullyConnectedLayer(1,"Name","fc_31")
    regressionLayer("Name","regressionoutput")];
lgraph = layerGraph(layers);
end

%% Create 4 layer network
if size(net_vec,2)==4
layers = [
    imageInputLayer([1,231],"Name","featureinput")
    fullyConnectedLayer(net_vec(1),"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(net_vec(2),"Name","fc_21")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(net_vec(3),"Name","fc_22")
    reluLayer("Name","relu_22")
    fullyConnectedLayer(net_vec(4),"Name","fc_31")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(1,"Name","fc_4")
    regressionLayer("Name","regressionoutput")];
lgraph = layerGraph(layers);
end


miniBatchSize  = 32;
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Xval,...
    'ValidationFrequency',30, ...
    'ValidationPatience',10, ...
    'Plots','none', ...
    'Verbose',true , ...
    'ExecutionEnvironment','cpu');
%plot(lgraph);
[net,info] = trainNetwork(Xtrain,lgraph,options);
