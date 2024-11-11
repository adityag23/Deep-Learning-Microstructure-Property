function [tempLayers] = build_compbranch()
tempLayers = [
    imageInputLayer([1,231],"Name","featureinput")
    fullyConnectedLayer(500,"Name","fc_1_feat")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(2000,"Name","fc_2_feat")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(2000,"Name","fc_3_feat")
    reluLayer("Name","relu_22")
    fullyConnectedLayer(500,"Name","fc_4_feat")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(1,"Name","fc_out_feat")];
end