function [ntrain,nval]= gen_rand_indices(n,pv)
nv = round(n*pv/100);
nt = n-nv;
ntrain = zeros(nt,1);
norg = 1:n;
nval = randperm(n,nv);
k=1;
for i=1:nv
for j=1:n
if norg(j) == nval(i)
norg(j) = 0;
end
end
end
for i=1:n
    if norg(i)~=0
        ntrain(k)=norg(i);
        k=k+1;
    end
end
end