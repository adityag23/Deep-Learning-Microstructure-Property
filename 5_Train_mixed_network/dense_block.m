function dense_layers = dense_block(lgraph,bnum,num_blocks)
size1=[1 1];
size2 = [3 3];
nfilt1 = 128;
nfilt2 = 32;

for i = 1:num_blocks
    name = strcat(bnum,"_conv_block_",string(i));
    conv_layers{i} = conv_block(size1,nfilt1,size2,nfilt2,name);
    concat_layer{i} = depthConcatenationLayer(2,"Name",strcat(name,"_concat"));
    lgraph = addLayers(lgraph,conv_layers{i}); 
    lgraph = addLayers(lgraph,concat_layer{i});
    lgraph = connectLayers(lgraph,conv_layers{i}(end).Name,strcat(concat_layer{i}.Name,"/in1"));
    if i==1 && bnum == "1"
        lgraph = connectLayers(lgraph,"pool1",strcat(concat_layer{i}.Name,"/in2"));
        lgraph = connectLayers(lgraph,"pool1",conv_layers{i}(1).Name);
    end
end
for i=1:num_blocks-1
    lgraph = connectLayers(lgraph,concat_layer{i}.Name,strcat(concat_layer{i+1}.Name,"/in2"));
    lgraph = connectLayers(lgraph,concat_layer{i}.Name,strcat(conv_layers{i+1}(1).Name));
end
dense_layers = lgraph;
end
