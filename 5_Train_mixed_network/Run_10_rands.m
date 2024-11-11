for i=1:3
    [net,info]=trainnetwork()
    rands3{1,i}.net=net;
    rands3{1,i}.info=info;
end
