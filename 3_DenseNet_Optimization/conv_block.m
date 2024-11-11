function conv_layers = conv_block(size1,nfilt1,size2,nfilt2,name)
conv_layers = [
    batchNormalizationLayer("Name",strcat(name,"_","bn1"))
    reluLayer("Name",strcat(name,"_","relu1"))
    convolution2dLayer(size1,nfilt1,"Name",strcat(name,"_","conv1"),"BiasLearnRateFactor",0)
    batchNormalizationLayer("Name",strcat(name,"_","bn2"))
    reluLayer("Name",strcat(name,"_","relu2"))
    convolution2dLayer(size2,nfilt2,"Name",strcat(name,"_","conv2"),"BiasLearnRateFactor",0,"Padding","same")];
end