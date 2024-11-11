function [hardness,h2,x,y,ims,af,af2,sp]=hsurf(net,mean_fv) 
c=1;
d=1;
e=1;
for sz=8:4:56
    d=1;
    for np=3:2:12
    npt=np*np;
    [im,af(c,d),sp(c,d),cond]=gen_im_primary2(sz,npt);
    hardness(c,d)=predict(net,im,mean_fv);
    h2(e)=hardness(c,d);
    ims{c,d}=im;
    x(e)=sz;
    y(e)=npt;
    af2(e)=af(c,d);
    d=d+1;
    e=e+1;
    end
    c=c+1;
end
