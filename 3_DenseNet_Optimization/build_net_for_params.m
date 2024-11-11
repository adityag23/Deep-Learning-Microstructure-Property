function [network] = build_net_for_params(num_conv,tcn) 
%num_conv is an array that specifies number of convolution blocks per dense block 
%example: num_conv = [1 2 3] creates 3 dense blocks with 1,2 and 3 conv_blocks each
num_dense = size(num_conv,2);
lgraph = layerGraph();
%% Input Layers
tempLayers = [
    imageInputLayer([224 224 3],"Name","input","Mean",155,"StandardDeviation",10,"Normalization","zscore")
    convolution2dLayer([7 7],64,"Name","first_conv","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","first_bn")
    reluLayer("Name","first_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])]; 
lgraph = addLayers(lgraph,tempLayers);

%% Dense Layers
k=32;
for i=1:num_dense
    lgraph = dense_block(lgraph,string(i),num_conv(i));
    previous_concat = lgraph.Layers(end).Name;
    if i<num_dense
    tcn=(tcn+k*num_conv(i))*0.5; %32 growth rate 0.5 compression factor
    tblock = trans_block(string(i),tcn);
    lgraph = addLayers(lgraph,tblock);
    lgraph = connectLayers(lgraph,previous_concat,tblock(1).Name);
    end
end

for i=2:num_dense
    lgraph = connectLayers(lgraph,strcat(string(i-1),"_trans_pool"),strcat(string(i),"_conv_block_1_bn1"));
    lgraph = connectLayers(lgraph,strcat(string(i-1),"_trans_pool"),strcat(string(i),"_conv_block_1_concat/in2"));
end

%% Output Layers
tempLayers = [
    batchNormalizationLayer("Name","last_bn")
    globalAveragePooling2dLayer("Name","last_avg_pool")
    fullyConnectedLayer(1,"Name","last_fc1")
    %regressionLayer("Name","regressionoutput")
    ];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,previous_concat,"last_bn");
network = lgraph;
end
