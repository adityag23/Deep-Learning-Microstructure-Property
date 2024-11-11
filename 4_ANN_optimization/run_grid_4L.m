init_net_4 = [500,600,600,500]
c=1;

for i=500:500
    for j=600:200:2000
	for k=600:200:2000
	    for l=500:500
            net_vec=[i,j,k,l];
            grid_4l{c}=net_vec;
	    [net,info]=create_train_networks(Xtrain,Xval,net_vec);
            result2{1,c}.net=net;
            result2{1,c}.info=info;
            valrmse2(c) = info.FinalValidationRMSE;
            c=c+1;
            end
	end
    end
end

