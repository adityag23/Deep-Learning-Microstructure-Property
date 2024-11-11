function lgraph = create_d201()
lgraph = layerGraph();
tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Normalization","zscore")
    convolution2dLayer([7 7],64,"Name","conv1|conv","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","conv1|bn")
    reluLayer("Name","conv1|relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block1_0_bn")
    reluLayer("Name","conv2_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block1_1_bn")
    reluLayer("Name","conv2_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block2_0_bn")
    reluLayer("Name","conv2_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block2_1_bn")
    reluLayer("Name","conv2_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block3_0_bn")
    reluLayer("Name","conv2_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block3_1_bn")
    reluLayer("Name","conv2_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block4_0_bn")
    reluLayer("Name","conv2_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block4_1_bn")
    reluLayer("Name","conv2_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block5_0_bn")
    reluLayer("Name","conv2_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block5_1_bn")
    reluLayer("Name","conv2_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block6_0_bn")
    reluLayer("Name","conv2_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block6_1_bn")
    reluLayer("Name","conv2_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv2_block6_concat")
    batchNormalizationLayer("Name","pool2_bn")
    reluLayer("Name","pool2_relu")
    convolution2dLayer([1 1],128,"Name","pool2_conv","BiasLearnRateFactor",0)
    averagePooling2dLayer([2 2],"Name","pool2_pool","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block1_0_bn")
    reluLayer("Name","conv3_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block1_1_bn")
    reluLayer("Name","conv3_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block2_0_bn")
    reluLayer("Name","conv3_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block2_1_bn")
    reluLayer("Name","conv3_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block3_0_bn")
    reluLayer("Name","conv3_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block3_1_bn")
    reluLayer("Name","conv3_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block4_0_bn")
    reluLayer("Name","conv3_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block4_1_bn")
    reluLayer("Name","conv3_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block5_0_bn")
    reluLayer("Name","conv3_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block5_1_bn")
    reluLayer("Name","conv3_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block6_0_bn")
    reluLayer("Name","conv3_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block6_1_bn")
    reluLayer("Name","conv3_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block6_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block7_0_bn")
    reluLayer("Name","conv3_block7_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block7_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block7_1_bn")
    reluLayer("Name","conv3_block7_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block7_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block7_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block8_0_bn")
    reluLayer("Name","conv3_block8_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block8_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block8_1_bn")
    reluLayer("Name","conv3_block8_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block8_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block8_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block9_0_bn")
    reluLayer("Name","conv3_block9_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block9_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block9_1_bn")
    reluLayer("Name","conv3_block9_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block9_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block9_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block10_0_bn")
    reluLayer("Name","conv3_block10_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block10_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block10_1_bn")
    reluLayer("Name","conv3_block10_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block10_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block10_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block11_0_bn")
    reluLayer("Name","conv3_block11_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block11_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block11_1_bn")
    reluLayer("Name","conv3_block11_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block11_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block11_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block12_0_bn")
    reluLayer("Name","conv3_block12_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block12_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block12_1_bn")
    reluLayer("Name","conv3_block12_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block12_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv3_block12_concat")
    batchNormalizationLayer("Name","pool3_bn")
    reluLayer("Name","pool3_relu")
    convolution2dLayer([1 1],256,"Name","pool3_conv","BiasLearnRateFactor",0)
    averagePooling2dLayer([2 2],"Name","pool3_pool","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block1_0_bn")
    reluLayer("Name","conv4_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block1_1_bn")
    reluLayer("Name","conv4_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block2_0_bn")
    reluLayer("Name","conv4_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block2_1_bn")
    reluLayer("Name","conv4_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block3_0_bn")
    reluLayer("Name","conv4_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block3_1_bn")
    reluLayer("Name","conv4_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block4_0_bn")
    reluLayer("Name","conv4_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block4_1_bn")
    reluLayer("Name","conv4_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block5_0_bn")
    reluLayer("Name","conv4_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block5_1_bn")
    reluLayer("Name","conv4_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block6_0_bn")
    reluLayer("Name","conv4_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block6_1_bn")
    reluLayer("Name","conv4_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block6_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block7_0_bn")
    reluLayer("Name","conv4_block7_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block7_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block7_1_bn")
    reluLayer("Name","conv4_block7_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block7_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block7_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block8_0_bn")
    reluLayer("Name","conv4_block8_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block8_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block8_1_bn")
    reluLayer("Name","conv4_block8_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block8_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block8_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block9_0_bn")
    reluLayer("Name","conv4_block9_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block9_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block9_1_bn")
    reluLayer("Name","conv4_block9_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block9_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block9_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block10_0_bn")
    reluLayer("Name","conv4_block10_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block10_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block10_1_bn")
    reluLayer("Name","conv4_block10_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block10_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block10_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block11_0_bn")
    reluLayer("Name","conv4_block11_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block11_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block11_1_bn")
    reluLayer("Name","conv4_block11_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block11_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block11_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block12_0_bn")
    reluLayer("Name","conv4_block12_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block12_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block12_1_bn")
    reluLayer("Name","conv4_block12_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block12_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block12_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block13_0_bn")
    reluLayer("Name","conv4_block13_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block13_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block13_1_bn")
    reluLayer("Name","conv4_block13_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block13_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block13_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block14_0_bn")
    reluLayer("Name","conv4_block14_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block14_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block14_1_bn")
    reluLayer("Name","conv4_block14_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block14_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block14_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block15_0_bn")
    reluLayer("Name","conv4_block15_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block15_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block15_1_bn")
    reluLayer("Name","conv4_block15_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block15_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block15_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block16_0_bn")
    reluLayer("Name","conv4_block16_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block16_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block16_1_bn")
    reluLayer("Name","conv4_block16_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block16_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block16_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block17_0_bn")
    reluLayer("Name","conv4_block17_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block17_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block17_1_bn")
    reluLayer("Name","conv4_block17_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block17_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block17_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block18_0_bn")
    reluLayer("Name","conv4_block18_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block18_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block18_1_bn")
    reluLayer("Name","conv4_block18_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block18_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block18_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block19_0_bn")
    reluLayer("Name","conv4_block19_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block19_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block19_1_bn")
    reluLayer("Name","conv4_block19_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block19_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block19_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block20_0_bn")
    reluLayer("Name","conv4_block20_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block20_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block20_1_bn")
    reluLayer("Name","conv4_block20_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block20_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block20_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block21_0_bn")
    reluLayer("Name","conv4_block21_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block21_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block21_1_bn")
    reluLayer("Name","conv4_block21_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block21_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block21_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block22_0_bn")
    reluLayer("Name","conv4_block22_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block22_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block22_1_bn")
    reluLayer("Name","conv4_block22_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block22_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block22_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block23_0_bn")
    reluLayer("Name","conv4_block23_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block23_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block23_1_bn")
    reluLayer("Name","conv4_block23_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block23_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block23_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block24_0_bn")
    reluLayer("Name","conv4_block24_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block24_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block24_1_bn")
    reluLayer("Name","conv4_block24_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block24_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block24_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block25_0_bn")
    reluLayer("Name","conv4_block25_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block25_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block25_1_bn")
    reluLayer("Name","conv4_block25_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block25_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block25_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block26_0_bn")
    reluLayer("Name","conv4_block26_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block26_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block26_1_bn")
    reluLayer("Name","conv4_block26_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block26_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block26_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block27_0_bn")
    reluLayer("Name","conv4_block27_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block27_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block27_1_bn")
    reluLayer("Name","conv4_block27_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block27_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block27_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block28_0_bn")
    reluLayer("Name","conv4_block28_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block28_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block28_1_bn")
    reluLayer("Name","conv4_block28_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block28_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block28_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block29_0_bn")
    reluLayer("Name","conv4_block29_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block29_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block29_1_bn")
    reluLayer("Name","conv4_block29_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block29_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block29_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block30_0_bn")
    reluLayer("Name","conv4_block30_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block30_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block30_1_bn")
    reluLayer("Name","conv4_block30_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block30_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block30_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block31_0_bn")
    reluLayer("Name","conv4_block31_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block31_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block31_1_bn")
    reluLayer("Name","conv4_block31_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block31_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block31_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block32_0_bn")
    reluLayer("Name","conv4_block32_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block32_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block32_1_bn")
    reluLayer("Name","conv4_block32_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block32_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block32_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block33_0_bn")
    reluLayer("Name","conv4_block33_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block33_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block33_1_bn")
    reluLayer("Name","conv4_block33_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block33_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block33_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block34_0_bn")
    reluLayer("Name","conv4_block34_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block34_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block34_1_bn")
    reluLayer("Name","conv4_block34_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block34_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block34_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block35_0_bn")
    reluLayer("Name","conv4_block35_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block35_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block35_1_bn")
    reluLayer("Name","conv4_block35_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block35_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block35_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block36_0_bn")
    reluLayer("Name","conv4_block36_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block36_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block36_1_bn")
    reluLayer("Name","conv4_block36_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block36_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block36_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block37_0_bn")
    reluLayer("Name","conv4_block37_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block37_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block37_1_bn")
    reluLayer("Name","conv4_block37_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block37_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block37_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block38_0_bn")
    reluLayer("Name","conv4_block38_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block38_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block38_1_bn")
    reluLayer("Name","conv4_block38_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block38_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block38_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block39_0_bn")
    reluLayer("Name","conv4_block39_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block39_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block39_1_bn")
    reluLayer("Name","conv4_block39_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block39_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block39_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block40_0_bn")
    reluLayer("Name","conv4_block40_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block40_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block40_1_bn")
    reluLayer("Name","conv4_block40_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block40_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block40_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block41_0_bn")
    reluLayer("Name","conv4_block41_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block41_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block41_1_bn")
    reluLayer("Name","conv4_block41_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block41_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block41_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block42_0_bn")
    reluLayer("Name","conv4_block42_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block42_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block42_1_bn")
    reluLayer("Name","conv4_block42_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block42_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block42_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block43_0_bn")
    reluLayer("Name","conv4_block43_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block43_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block43_1_bn")
    reluLayer("Name","conv4_block43_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block43_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block43_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block44_0_bn")
    reluLayer("Name","conv4_block44_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block44_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block44_1_bn")
    reluLayer("Name","conv4_block44_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block44_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block44_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block45_0_bn")
    reluLayer("Name","conv4_block45_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block45_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block45_1_bn")
    reluLayer("Name","conv4_block45_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block45_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block45_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block46_0_bn")
    reluLayer("Name","conv4_block46_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block46_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block46_1_bn")
    reluLayer("Name","conv4_block46_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block46_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block46_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block47_0_bn")
    reluLayer("Name","conv4_block47_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block47_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block47_1_bn")
    reluLayer("Name","conv4_block47_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block47_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block47_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block48_0_bn")
    reluLayer("Name","conv4_block48_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block48_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block48_1_bn")
    reluLayer("Name","conv4_block48_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block48_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv4_block48_concat")
    batchNormalizationLayer("Name","pool4_bn")
    reluLayer("Name","pool4_relu")
    convolution2dLayer([1 1],896,"Name","pool4_conv","BiasLearnRateFactor",0)
    averagePooling2dLayer([2 2],"Name","pool4_pool","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block1_0_bn")
    reluLayer("Name","conv5_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block1_1_bn")
    reluLayer("Name","conv5_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block2_0_bn")
    reluLayer("Name","conv5_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block2_1_bn")
    reluLayer("Name","conv5_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block3_0_bn")
    reluLayer("Name","conv5_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block3_1_bn")
    reluLayer("Name","conv5_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block4_0_bn")
    reluLayer("Name","conv5_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block4_1_bn")
    reluLayer("Name","conv5_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block5_0_bn")
    reluLayer("Name","conv5_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block5_1_bn")
    reluLayer("Name","conv5_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block6_0_bn")
    reluLayer("Name","conv5_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block6_1_bn")
    reluLayer("Name","conv5_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block6_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block7_0_bn")
    reluLayer("Name","conv5_block7_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block7_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block7_1_bn")
    reluLayer("Name","conv5_block7_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block7_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block7_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block8_0_bn")
    reluLayer("Name","conv5_block8_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block8_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block8_1_bn")
    reluLayer("Name","conv5_block8_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block8_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block8_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block9_0_bn")
    reluLayer("Name","conv5_block9_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block9_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block9_1_bn")
    reluLayer("Name","conv5_block9_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block9_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block9_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block10_0_bn")
    reluLayer("Name","conv5_block10_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block10_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block10_1_bn")
    reluLayer("Name","conv5_block10_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block10_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block10_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block11_0_bn")
    reluLayer("Name","conv5_block11_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block11_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block11_1_bn")
    reluLayer("Name","conv5_block11_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block11_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block11_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block12_0_bn")
    reluLayer("Name","conv5_block12_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block12_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block12_1_bn")
    reluLayer("Name","conv5_block12_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block12_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block12_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block13_0_bn")
    reluLayer("Name","conv5_block13_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block13_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block13_1_bn")
    reluLayer("Name","conv5_block13_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block13_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block13_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block14_0_bn")
    reluLayer("Name","conv5_block14_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block14_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block14_1_bn")
    reluLayer("Name","conv5_block14_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block14_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block14_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block15_0_bn")
    reluLayer("Name","conv5_block15_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block15_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block15_1_bn")
    reluLayer("Name","conv5_block15_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block15_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block15_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block16_0_bn")
    reluLayer("Name","conv5_block16_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block16_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block16_1_bn")
    reluLayer("Name","conv5_block16_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block16_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block16_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block17_0_bn")
    reluLayer("Name","conv5_block17_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block17_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block17_1_bn")
    reluLayer("Name","conv5_block17_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block17_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block17_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block18_0_bn")
    reluLayer("Name","conv5_block18_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block18_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block18_1_bn")
    reluLayer("Name","conv5_block18_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block18_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block18_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block19_0_bn")
    reluLayer("Name","conv5_block19_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block19_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block19_1_bn")
    reluLayer("Name","conv5_block19_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block19_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block19_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block20_0_bn")
    reluLayer("Name","conv5_block20_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block20_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block20_1_bn")
    reluLayer("Name","conv5_block20_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block20_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block20_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block21_0_bn")
    reluLayer("Name","conv5_block21_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block21_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block21_1_bn")
    reluLayer("Name","conv5_block21_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block21_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block21_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block22_0_bn")
    reluLayer("Name","conv5_block22_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block22_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block22_1_bn")
    reluLayer("Name","conv5_block22_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block22_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block22_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block23_0_bn")
    reluLayer("Name","conv5_block23_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block23_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block23_1_bn")
    reluLayer("Name","conv5_block23_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block23_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block23_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block24_0_bn")
    reluLayer("Name","conv5_block24_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block24_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block24_1_bn")
    reluLayer("Name","conv5_block24_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block24_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block24_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block25_0_bn")
    reluLayer("Name","conv5_block25_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block25_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block25_1_bn")
    reluLayer("Name","conv5_block25_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block25_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block25_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block26_0_bn")
    reluLayer("Name","conv5_block26_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block26_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block26_1_bn")
    reluLayer("Name","conv5_block26_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block26_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block26_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block27_0_bn")
    reluLayer("Name","conv5_block27_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block27_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block27_1_bn")
    reluLayer("Name","conv5_block27_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block27_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block27_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block28_0_bn")
    reluLayer("Name","conv5_block28_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block28_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block28_1_bn")
    reluLayer("Name","conv5_block28_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block28_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block28_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block29_0_bn")
    reluLayer("Name","conv5_block29_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block29_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block29_1_bn")
    reluLayer("Name","conv5_block29_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block29_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block29_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block30_0_bn")
    reluLayer("Name","conv5_block30_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block30_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block30_1_bn")
    reluLayer("Name","conv5_block30_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block30_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block30_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block31_0_bn")
    reluLayer("Name","conv5_block31_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block31_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block31_1_bn")
    reluLayer("Name","conv5_block31_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block31_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv5_block31_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv5_block32_0_bn")
    reluLayer("Name","conv5_block32_0_relu")
    convolution2dLayer([1 1],128,"Name","conv5_block32_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv5_block32_1_bn")
    reluLayer("Name","conv5_block32_1_relu")
    convolution2dLayer([3 3],32,"Name","conv5_block32_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv5_block32_concat")
    batchNormalizationLayer("Name","bn")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(1,"Name","fc1")
    regressionLayer("Name","output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"pool1","conv2_block1_0_bn");
lgraph = connectLayers(lgraph,"pool1","conv2_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block1_2_conv","conv2_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_0_bn");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block2_2_conv","conv2_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block2_concat","conv2_block3_0_bn");
lgraph = connectLayers(lgraph,"conv2_block2_concat","conv2_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block3_2_conv","conv2_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block3_concat","conv2_block4_0_bn");
lgraph = connectLayers(lgraph,"conv2_block3_concat","conv2_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block4_2_conv","conv2_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block4_concat","conv2_block5_0_bn");
lgraph = connectLayers(lgraph,"conv2_block4_concat","conv2_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block5_2_conv","conv2_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block5_concat","conv2_block6_0_bn");
lgraph = connectLayers(lgraph,"conv2_block5_concat","conv2_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block6_2_conv","conv2_block6_concat/in2");
lgraph = connectLayers(lgraph,"pool2_pool","conv3_block1_0_bn");
lgraph = connectLayers(lgraph,"pool2_pool","conv3_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block1_2_conv","conv3_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block1_concat","conv3_block2_0_bn");
lgraph = connectLayers(lgraph,"conv3_block1_concat","conv3_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block2_2_conv","conv3_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block2_concat","conv3_block3_0_bn");
lgraph = connectLayers(lgraph,"conv3_block2_concat","conv3_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block3_2_conv","conv3_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block3_concat","conv3_block4_0_bn");
lgraph = connectLayers(lgraph,"conv3_block3_concat","conv3_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block4_2_conv","conv3_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block4_concat","conv3_block5_0_bn");
lgraph = connectLayers(lgraph,"conv3_block4_concat","conv3_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block5_2_conv","conv3_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block5_concat","conv3_block6_0_bn");
lgraph = connectLayers(lgraph,"conv3_block5_concat","conv3_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block6_2_conv","conv3_block6_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block6_concat","conv3_block7_0_bn");
lgraph = connectLayers(lgraph,"conv3_block6_concat","conv3_block7_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block7_2_conv","conv3_block7_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block7_concat","conv3_block8_0_bn");
lgraph = connectLayers(lgraph,"conv3_block7_concat","conv3_block8_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block8_2_conv","conv3_block8_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block8_concat","conv3_block9_0_bn");
lgraph = connectLayers(lgraph,"conv3_block8_concat","conv3_block9_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block9_2_conv","conv3_block9_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block9_concat","conv3_block10_0_bn");
lgraph = connectLayers(lgraph,"conv3_block9_concat","conv3_block10_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block10_2_conv","conv3_block10_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block10_concat","conv3_block11_0_bn");
lgraph = connectLayers(lgraph,"conv3_block10_concat","conv3_block11_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block11_2_conv","conv3_block11_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block11_concat","conv3_block12_0_bn");
lgraph = connectLayers(lgraph,"conv3_block11_concat","conv3_block12_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block12_2_conv","conv3_block12_concat/in2");
lgraph = connectLayers(lgraph,"pool3_pool","conv4_block1_0_bn");
lgraph = connectLayers(lgraph,"pool3_pool","conv4_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block1_2_conv","conv4_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block1_concat","conv4_block2_0_bn");
lgraph = connectLayers(lgraph,"conv4_block1_concat","conv4_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block2_2_conv","conv4_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block2_concat","conv4_block3_0_bn");
lgraph = connectLayers(lgraph,"conv4_block2_concat","conv4_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block3_2_conv","conv4_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block3_concat","conv4_block4_0_bn");
lgraph = connectLayers(lgraph,"conv4_block3_concat","conv4_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block4_2_conv","conv4_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block4_concat","conv4_block5_0_bn");
lgraph = connectLayers(lgraph,"conv4_block4_concat","conv4_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block5_2_conv","conv4_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block5_concat","conv4_block6_0_bn");
lgraph = connectLayers(lgraph,"conv4_block5_concat","conv4_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block6_2_conv","conv4_block6_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block6_concat","conv4_block7_0_bn");
lgraph = connectLayers(lgraph,"conv4_block6_concat","conv4_block7_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block7_2_conv","conv4_block7_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block7_concat","conv4_block8_0_bn");
lgraph = connectLayers(lgraph,"conv4_block7_concat","conv4_block8_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block8_2_conv","conv4_block8_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block8_concat","conv4_block9_0_bn");
lgraph = connectLayers(lgraph,"conv4_block8_concat","conv4_block9_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block9_2_conv","conv4_block9_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block9_concat","conv4_block10_0_bn");
lgraph = connectLayers(lgraph,"conv4_block9_concat","conv4_block10_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block10_2_conv","conv4_block10_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block10_concat","conv4_block11_0_bn");
lgraph = connectLayers(lgraph,"conv4_block10_concat","conv4_block11_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block11_2_conv","conv4_block11_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block11_concat","conv4_block12_0_bn");
lgraph = connectLayers(lgraph,"conv4_block11_concat","conv4_block12_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block12_2_conv","conv4_block12_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block12_concat","conv4_block13_0_bn");
lgraph = connectLayers(lgraph,"conv4_block12_concat","conv4_block13_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block13_2_conv","conv4_block13_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block13_concat","conv4_block14_0_bn");
lgraph = connectLayers(lgraph,"conv4_block13_concat","conv4_block14_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block14_2_conv","conv4_block14_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block14_concat","conv4_block15_0_bn");
lgraph = connectLayers(lgraph,"conv4_block14_concat","conv4_block15_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block15_2_conv","conv4_block15_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block15_concat","conv4_block16_0_bn");
lgraph = connectLayers(lgraph,"conv4_block15_concat","conv4_block16_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block16_2_conv","conv4_block16_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block16_concat","conv4_block17_0_bn");
lgraph = connectLayers(lgraph,"conv4_block16_concat","conv4_block17_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block17_2_conv","conv4_block17_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block17_concat","conv4_block18_0_bn");
lgraph = connectLayers(lgraph,"conv4_block17_concat","conv4_block18_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block18_2_conv","conv4_block18_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block18_concat","conv4_block19_0_bn");
lgraph = connectLayers(lgraph,"conv4_block18_concat","conv4_block19_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block19_2_conv","conv4_block19_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block19_concat","conv4_block20_0_bn");
lgraph = connectLayers(lgraph,"conv4_block19_concat","conv4_block20_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block20_2_conv","conv4_block20_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block20_concat","conv4_block21_0_bn");
lgraph = connectLayers(lgraph,"conv4_block20_concat","conv4_block21_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block21_2_conv","conv4_block21_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block21_concat","conv4_block22_0_bn");
lgraph = connectLayers(lgraph,"conv4_block21_concat","conv4_block22_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block22_2_conv","conv4_block22_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block22_concat","conv4_block23_0_bn");
lgraph = connectLayers(lgraph,"conv4_block22_concat","conv4_block23_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block23_2_conv","conv4_block23_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block23_concat","conv4_block24_0_bn");
lgraph = connectLayers(lgraph,"conv4_block23_concat","conv4_block24_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block24_2_conv","conv4_block24_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block24_concat","conv4_block25_0_bn");
lgraph = connectLayers(lgraph,"conv4_block24_concat","conv4_block25_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block25_2_conv","conv4_block25_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block25_concat","conv4_block26_0_bn");
lgraph = connectLayers(lgraph,"conv4_block25_concat","conv4_block26_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block26_2_conv","conv4_block26_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block26_concat","conv4_block27_0_bn");
lgraph = connectLayers(lgraph,"conv4_block26_concat","conv4_block27_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block27_2_conv","conv4_block27_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block27_concat","conv4_block28_0_bn");
lgraph = connectLayers(lgraph,"conv4_block27_concat","conv4_block28_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block28_2_conv","conv4_block28_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block28_concat","conv4_block29_0_bn");
lgraph = connectLayers(lgraph,"conv4_block28_concat","conv4_block29_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block29_2_conv","conv4_block29_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block29_concat","conv4_block30_0_bn");
lgraph = connectLayers(lgraph,"conv4_block29_concat","conv4_block30_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block30_2_conv","conv4_block30_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block30_concat","conv4_block31_0_bn");
lgraph = connectLayers(lgraph,"conv4_block30_concat","conv4_block31_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block31_2_conv","conv4_block31_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block31_concat","conv4_block32_0_bn");
lgraph = connectLayers(lgraph,"conv4_block31_concat","conv4_block32_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block32_2_conv","conv4_block32_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block32_concat","conv4_block33_0_bn");
lgraph = connectLayers(lgraph,"conv4_block32_concat","conv4_block33_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block33_2_conv","conv4_block33_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block33_concat","conv4_block34_0_bn");
lgraph = connectLayers(lgraph,"conv4_block33_concat","conv4_block34_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block34_2_conv","conv4_block34_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block34_concat","conv4_block35_0_bn");
lgraph = connectLayers(lgraph,"conv4_block34_concat","conv4_block35_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block35_2_conv","conv4_block35_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block35_concat","conv4_block36_0_bn");
lgraph = connectLayers(lgraph,"conv4_block35_concat","conv4_block36_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block36_2_conv","conv4_block36_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block36_concat","conv4_block37_0_bn");
lgraph = connectLayers(lgraph,"conv4_block36_concat","conv4_block37_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block37_2_conv","conv4_block37_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block37_concat","conv4_block38_0_bn");
lgraph = connectLayers(lgraph,"conv4_block37_concat","conv4_block38_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block38_2_conv","conv4_block38_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block38_concat","conv4_block39_0_bn");
lgraph = connectLayers(lgraph,"conv4_block38_concat","conv4_block39_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block39_2_conv","conv4_block39_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block39_concat","conv4_block40_0_bn");
lgraph = connectLayers(lgraph,"conv4_block39_concat","conv4_block40_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block40_2_conv","conv4_block40_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block40_concat","conv4_block41_0_bn");
lgraph = connectLayers(lgraph,"conv4_block40_concat","conv4_block41_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block41_2_conv","conv4_block41_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block41_concat","conv4_block42_0_bn");
lgraph = connectLayers(lgraph,"conv4_block41_concat","conv4_block42_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block42_2_conv","conv4_block42_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block42_concat","conv4_block43_0_bn");
lgraph = connectLayers(lgraph,"conv4_block42_concat","conv4_block43_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block43_2_conv","conv4_block43_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block43_concat","conv4_block44_0_bn");
lgraph = connectLayers(lgraph,"conv4_block43_concat","conv4_block44_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block44_2_conv","conv4_block44_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block44_concat","conv4_block45_0_bn");
lgraph = connectLayers(lgraph,"conv4_block44_concat","conv4_block45_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block45_2_conv","conv4_block45_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block45_concat","conv4_block46_0_bn");
lgraph = connectLayers(lgraph,"conv4_block45_concat","conv4_block46_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block46_2_conv","conv4_block46_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block46_concat","conv4_block47_0_bn");
lgraph = connectLayers(lgraph,"conv4_block46_concat","conv4_block47_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block47_2_conv","conv4_block47_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block47_concat","conv4_block48_0_bn");
lgraph = connectLayers(lgraph,"conv4_block47_concat","conv4_block48_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block48_2_conv","conv4_block48_concat/in2");
lgraph = connectLayers(lgraph,"pool4_pool","conv5_block1_0_bn");
lgraph = connectLayers(lgraph,"pool4_pool","conv5_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block1_2_conv","conv5_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block1_concat","conv5_block2_0_bn");
lgraph = connectLayers(lgraph,"conv5_block1_concat","conv5_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block2_2_conv","conv5_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block2_concat","conv5_block3_0_bn");
lgraph = connectLayers(lgraph,"conv5_block2_concat","conv5_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block3_2_conv","conv5_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block3_concat","conv5_block4_0_bn");
lgraph = connectLayers(lgraph,"conv5_block3_concat","conv5_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block4_2_conv","conv5_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block4_concat","conv5_block5_0_bn");
lgraph = connectLayers(lgraph,"conv5_block4_concat","conv5_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block5_2_conv","conv5_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block5_concat","conv5_block6_0_bn");
lgraph = connectLayers(lgraph,"conv5_block5_concat","conv5_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block6_2_conv","conv5_block6_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block6_concat","conv5_block7_0_bn");
lgraph = connectLayers(lgraph,"conv5_block6_concat","conv5_block7_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block7_2_conv","conv5_block7_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block7_concat","conv5_block8_0_bn");
lgraph = connectLayers(lgraph,"conv5_block7_concat","conv5_block8_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block8_2_conv","conv5_block8_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block8_concat","conv5_block9_0_bn");
lgraph = connectLayers(lgraph,"conv5_block8_concat","conv5_block9_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block9_2_conv","conv5_block9_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block9_concat","conv5_block10_0_bn");
lgraph = connectLayers(lgraph,"conv5_block9_concat","conv5_block10_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block10_2_conv","conv5_block10_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block10_concat","conv5_block11_0_bn");
lgraph = connectLayers(lgraph,"conv5_block10_concat","conv5_block11_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block11_2_conv","conv5_block11_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block11_concat","conv5_block12_0_bn");
lgraph = connectLayers(lgraph,"conv5_block11_concat","conv5_block12_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block12_2_conv","conv5_block12_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block12_concat","conv5_block13_0_bn");
lgraph = connectLayers(lgraph,"conv5_block12_concat","conv5_block13_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block13_2_conv","conv5_block13_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block13_concat","conv5_block14_0_bn");
lgraph = connectLayers(lgraph,"conv5_block13_concat","conv5_block14_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block14_2_conv","conv5_block14_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block14_concat","conv5_block15_0_bn");
lgraph = connectLayers(lgraph,"conv5_block14_concat","conv5_block15_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block15_2_conv","conv5_block15_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block15_concat","conv5_block16_0_bn");
lgraph = connectLayers(lgraph,"conv5_block15_concat","conv5_block16_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block16_2_conv","conv5_block16_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block16_concat","conv5_block17_0_bn");
lgraph = connectLayers(lgraph,"conv5_block16_concat","conv5_block17_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block17_2_conv","conv5_block17_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block17_concat","conv5_block18_0_bn");
lgraph = connectLayers(lgraph,"conv5_block17_concat","conv5_block18_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block18_2_conv","conv5_block18_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block18_concat","conv5_block19_0_bn");
lgraph = connectLayers(lgraph,"conv5_block18_concat","conv5_block19_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block19_2_conv","conv5_block19_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block19_concat","conv5_block20_0_bn");
lgraph = connectLayers(lgraph,"conv5_block19_concat","conv5_block20_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block20_2_conv","conv5_block20_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block20_concat","conv5_block21_0_bn");
lgraph = connectLayers(lgraph,"conv5_block20_concat","conv5_block21_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block21_2_conv","conv5_block21_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block21_concat","conv5_block22_0_bn");
lgraph = connectLayers(lgraph,"conv5_block21_concat","conv5_block22_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block22_2_conv","conv5_block22_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block22_concat","conv5_block23_0_bn");
lgraph = connectLayers(lgraph,"conv5_block22_concat","conv5_block23_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block23_2_conv","conv5_block23_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block23_concat","conv5_block24_0_bn");
lgraph = connectLayers(lgraph,"conv5_block23_concat","conv5_block24_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block24_2_conv","conv5_block24_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block24_concat","conv5_block25_0_bn");
lgraph = connectLayers(lgraph,"conv5_block24_concat","conv5_block25_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block25_2_conv","conv5_block25_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block25_concat","conv5_block26_0_bn");
lgraph = connectLayers(lgraph,"conv5_block25_concat","conv5_block26_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block26_2_conv","conv5_block26_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block26_concat","conv5_block27_0_bn");
lgraph = connectLayers(lgraph,"conv5_block26_concat","conv5_block27_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block27_2_conv","conv5_block27_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block27_concat","conv5_block28_0_bn");
lgraph = connectLayers(lgraph,"conv5_block27_concat","conv5_block28_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block28_2_conv","conv5_block28_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block28_concat","conv5_block29_0_bn");
lgraph = connectLayers(lgraph,"conv5_block28_concat","conv5_block29_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block29_2_conv","conv5_block29_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block29_concat","conv5_block30_0_bn");
lgraph = connectLayers(lgraph,"conv5_block29_concat","conv5_block30_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block30_2_conv","conv5_block30_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block30_concat","conv5_block31_0_bn");
lgraph = connectLayers(lgraph,"conv5_block30_concat","conv5_block31_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block31_2_conv","conv5_block31_concat/in2");
lgraph = connectLayers(lgraph,"conv5_block31_concat","conv5_block32_0_bn");
lgraph = connectLayers(lgraph,"conv5_block31_concat","conv5_block32_concat/in1");
lgraph = connectLayers(lgraph,"conv5_block32_2_conv","conv5_block32_concat/in2");

