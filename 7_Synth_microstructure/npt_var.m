function [hardness,x,ims,af,sp]=npt_var(net,mean_fv,sz) 
c=1;
for np=2:1:11
    npt=np*np;
    [im,af(c),sp(c),sd,rd,ard]=gen_im_primary(npt,sz);
    hardness(c)=predict(net,im,mean_fv);
    ims{c}=im;
    x(c)=npt;
    c=c+1;
end
