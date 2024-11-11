function[objective]=optimized_function(x)
numconv = [x.a1 x.a2 x.a3 x.a4];
tcn=512; %number of fixed transition filters
lgraph=build_dense_net(numconv,tcn);
[net,info] = train_for_optimization(lgraph);
objective=info.FinalValidationRMSE;
end
