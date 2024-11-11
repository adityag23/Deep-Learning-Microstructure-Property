function calc_net_params(results)
%supply bayesian optimization results structure to count number of parameters
for i=1:30
    nc1 = results_bo2.XTrace.a1(i);
    nc2 = results_bo2.XTrace.a2(i);
    nc3 = results_bo2.XTrace.a3(i);
    nc4 = results_bo2.XTrace.a4(i);
    net = build_net_for_paramas([nc1,nc2,nc3,nc4],512);
    par(i)=countparam(net);
end
