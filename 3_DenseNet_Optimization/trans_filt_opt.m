function[results_tcn,rmse_vals] = trans_filt_opt()
% optimization script for number of transition filters (tcn)
k=0; % from previous growth rate optimization where tcn=64
tcns = [32,64,128,256,512]; 
for i=1:5
[lgraph] = build_dense_net(num_conv,tcns(i),k);
[net,info]=train_for_optimization(lgraph);
results_tcn{1,i}.net=net;
results_tcn{1,i}.info=info;
rmse_vals(i) = info.FinalValidationRMSE
end



