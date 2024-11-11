function[results]=optimize_network()
%Defined dense block ranges not exceeding DenseNet201
a1 = optimizableVariable('a1',[1 6],'Type','integer');
a2 = optimizableVariable('a2',[1 12],'Type','integer');
a3 = optimizableVariable('a3',[1 48],'Type','integer');
a4 = optimizableVariable('a4',[1 32],'Type','integer');
y=table();
y.a1=1;
y.a2=1;
y.a3=1;
y.a4=1;
func = @(x)opfunc(x);
results = bayesopt(func,[a1,a2,a3,a4],'Verbose',1,'MaxObjectiveEvaluations',30,'NumSeedPoints',1,'InitialX',y);
end
