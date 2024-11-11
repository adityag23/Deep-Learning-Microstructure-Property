function [hardness,x,ims,af,sp]=size_var2(net,mean_fv,npt) 
c=1;
sz=2;
%cond=1;
while sz<=20
    [im,af(c),sp(c),cond]=gen_im_primary2(sz,npt);
    hardness(c)=predict(net,im,mean_fv);
    ims{c}=im;
    x(c)=sz;
    c=c+1;
    sz=sz+1;
end
