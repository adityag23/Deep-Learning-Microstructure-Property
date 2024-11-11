function[results_k,rmse_vals] = growth_rate_opt()
tcn=64 %from densenet201
ks = [0,4,8,12,16,32]
for i=1:6
[lgraph] = build_dense_net(num_conv,tcn,k(i));
[net,info]=train_for_optimization(lgraph);
results_k{1,i}.net=net;
results_k{1,i}.info=info;
rmse_vals(i) = info.FinalValidationRMSE
end



