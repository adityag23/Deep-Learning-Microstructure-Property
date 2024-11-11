function [net,info]=train_for_optimization(lgraph)
[Xtrain,Xval] = create_datastores();

%% Fixed hyper-parameters
miniBatchSize  = 32;
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Xval,...
    'ValidationFrequency',30, ...
    'ValidationPatience',10, ...
    'Plots','none', ...
    'Verbose',true , ...
    'ExecutionEnvironment','gpu');
[net,info] = trainNetwork(Xtrain,lgraph,options);