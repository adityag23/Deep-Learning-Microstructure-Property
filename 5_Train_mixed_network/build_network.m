function [network] = build_network(num_conv) 
%% Build branches
lgraphi = build_imbranch(num_conv);
lgraphc = build_compbranch();
%% Output layers
lgrapho = [
    concatenationLayer(1,2,"Name","concat_out")
    fullyConnectedLayer(1,"Name","fc_out")
    regressionLayer("Name","regressionoutput")];
%% Add layers
network = addLayers(lgraphi,lgraphc);
network = addLayers(network,lgrapho);
%% Connect layers
network = connectLayers(network,"last_fc1","concat_out/in1");
network = connectLayers(network,"fc_out_feat","concat_out/in2");
end