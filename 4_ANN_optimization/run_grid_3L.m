init_net_3 = [300,600,300]
c=1;

for i=300:100:500
    for j=600:200:2000
	for k=300:100:500
            net_vec=[i,j,k];
            grid_3l(c)="["+net_vec(1)+","+net_vec(2)+","+net_vec(3)+"]";
	    %[net,info]=create_train_networks(Xtrain,Xval,net_vec);
        %    result{1,c}.net=net;
        %    result{1,c}.info=info;
        %    valrmse(c) = info.FinalValidationRMSE;
            c=c+1;
	end
    end
end

