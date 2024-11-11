yt = info.TrainingRMSE;
x = 1:size(yt,2);
y = info.ValidationRMSE;
n=size(y);
xs = x(~isnan(y));
ys = y(~isnan(y));
yi = interp1(xs,ys,x,'linear');

%figure
%hold
plot(yt)
plot(xs,ys)

