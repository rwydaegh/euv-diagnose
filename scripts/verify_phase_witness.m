addpath(genpath('research/sources/sherwin'));
m=load('research/results/sherwin-input-131.mat');
for field={'N_ML','n_cap','n_per','n_contam','n_etch'}
    m.(field{1})=double(m.(field{1}));
end
for name={'probe','tight'}
    d=load(['research/results/phase-ambiguity-' name{1} '-input.mat']);x=d.x;
    [a,b]=calcMLAbsEtchFresnel(m.N_ML,m.n_cap,m.n_per,m.n_contam,m.n_etch, ...
        x(m.ind_struct.thick),m.amu,m.rho_nom,x(m.ind_struct.rough), ...
        m.f0f1_elements_r,reshape(x(m.ind_struct.composition),length(m.amu),[]), ...
        m.lambdaTheta,m.pol,1,x(m.ind_struct.substrate_roughness));
    save('-mat7-binary',['research/results/phase-ambiguity-' name{1} '-reference.mat'],'a','b');
end
