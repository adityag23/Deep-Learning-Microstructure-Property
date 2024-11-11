function  [im1,af,spacing,cond] = gen_im_primary2(size,npart)
%npart is number of precipitates: to be prefect squares only
%afract is the area feaction of the precipitate phase
%r,a,b are superellipsoid parameters
rs=[4,4];
a=1;
b=1;
x=1;
y=1;
fore = 255;
back = 0;
%npart = (afract*224*224/(size*size));
npt = round(sqrt(npart));
noise_v=0;
size_v=0;
a_v=0;
b_v=0;


size=round(size/2);

if mod(npt,2)~=0
step = round(224/(npt));
else
step = round(112/(0.5*npt));
end

mat = back*ones(112)+noise_v.*(-1+2.*rand(112));
p=1;

if mod(npt,2)==0
npt_red = npt/2;
a=step/(npt);
 if round(step/2)>size
 cond=1;
 else
 cond=0;
 end
else
npt_red = round(npt/2+1);
 if step-size>0
 cond=1;
 else
 cond=0;
 end
end

for i=1:npt_red

    if i~=1  
    centerx(i) = step+centerx(i-1);
    elseif mod(npt,2)~=0 && i==1
    centerx(i)=1;
    else
    centerx(i)=round(step/2);
    end

    for j=1:npt_red
        s = randi([size-size_v,size+size_v],1,1);
        r = rs(randi([1,2],1,1));
        a = 1+a_v*rand();
        b = 1+b_v*rand();

        sd(p)=s;
        rd(p)=r;
        ard(p)=a/b;
        
        if j~=1 
        centery(j) = step+centery(j-1);
        elseif mod(npt,2)~=0 && j==1
        centery(j) = 1;
        else
        centery(j) = round(step/2);
        end

        x=1;
        for l=1:112
            for m=1:112
                if ((l-centerx(i))/a)^r+((m-centery(j))/b)^r<s^r
                mat(x,y)=fore+noise_v*(-1+2*rand());
                end
                y=y+1;
            end
            x=x+1;
            y=1;
        end
        p=p+1;
    end
end

fl_1=fliplr(mat);
matl = zeros(112,224);
matl(1:112,1:112)=fl_1;
matl(1:112,113:224)=mat;

fl_v=flip(matl);
mat_full = zeros(224,224);
mat_full(1:112,1:224)=fl_v;
mat_full(113:224,1:224)=matl;

%image(fl_1);

ppt_pix=0;
for i=1:112
    for j=1:112
    	if mat(i,j)>=fore-noise_v
           ppt_pix=ppt_pix+1;
        end
    end
end

af = ppt_pix/(112*112);
spacing = (224-(npt*2*size))/(npt);

im = uint8(mat_full);
im1 = cat(3,im,im,im);
imshow(im1)
