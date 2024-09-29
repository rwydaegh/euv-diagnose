addpath(genpath('research/sources/sherwin'));
for tagcell={'131','073','074'}
    tag=tagcell{1}; m=load(['research/results/sherwin-input-' tag '.mat']);
    for field={'N_ML','n_cap','n_per','n_contam','n_etch'}
        m.(field{1})=double(m.(field{1}));
    end
    for kind={'s','p'}
        pol=kind{1}; x=m.x;
        x(m.ind_struct.thick(1)) += 0.03;
        x(m.ind_struct.thick(end)) += 0.07;
        x(m.ind_struct.rough) += 0.02;
        grid=m.lambdaTheta; grid(:,2) += 0.03;
        [a,b]=calcMLAbsEtchFresnel(m.N_ML,m.n_cap,m.n_per,m.n_contam,m.n_etch, ...
            x(m.ind_struct.thick),m.amu,m.rho_nom,x(m.ind_struct.rough), ...
            m.f0f1_elements_r,reshape(x(m.ind_struct.composition),length(m.amu),[]), ...
            grid,pol,1,x(m.ind_struct.substrate_roughness));
        save('-mat7-binary',['research/results/optics-reference-' tag '-' pol '.mat'],'a','b');
    end
end
