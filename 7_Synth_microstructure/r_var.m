function [hardness,x,ims,af,sp]=r_var(net,mean_fv,npt,sz) 
c=1;
r=2;
cond=1;
while r<=12
    [im,af(c),sp(c),cond]=gen_im_primary3(sz,npt,r);
    hardness(c)=predict(net,im,mean_fv);
    ims{c}=im;
    x(c)=r;
    c=c+1;
    r=r+2;
end
