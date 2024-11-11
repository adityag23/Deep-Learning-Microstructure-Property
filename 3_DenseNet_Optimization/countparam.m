function le = countparam(network)
%[network]=build_dense_net([6,12,48,32]);
dln = dlnetwork(network);
lbs = dln.Learnables.Value;
le = 0;
for i=1:size(lbs,1)
le = le + prod(size(lbs{i}));
end