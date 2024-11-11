function transition = trans_block(name,tcn)
tcsize = [1 1];
tpsize = [2 2];
%tcn = 128;
transition = [
     batchNormalizationLayer("Name",strcat(name,"_","trans_bn"))
     reluLayer("Name",strcat(name,"_","trans_relu"))
     convolution2dLayer(tcsize,tcn,"Name",strcat(name,"_","trans_conv"),"BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
     averagePooling2dLayer(tpsize,"Name",strcat(name,"_","trans_pool"),"Padding",[1 1 1 1],"Stride",[2 2])];
end