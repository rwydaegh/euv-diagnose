function [r_MLabs,r_MLetch,r_ML,r_abs,t_abs,r_etch,t_etch,M_ML,M_abs,M_etch,M_MLabs,M_MLetch] = calcMLAbsEtchFresnel(N_ML,n_cap,n_per,n_contam,n_etch,thick,amu,rho_nom,rough,f0f1_elements,composition,lambdaTheta,pol,scale_factor,correlated_roughness)
    if nargin < 14
        scale_factor = ones(1,size(thick,2));
    end
    if nargin < 15
        correlated_roughness = zeros(1,size(thick,2));
    end
    N = size(thick,2);
    f = @(x) reshape(x,[],N);
    out = mkMLAbsEtch(n_cap,n_per,n_etch,thick,amu,rho_nom,rough,f0f1_elements,composition,lambdaTheta);
    layer_thick_ML = out.mirror.thick;
    layer_nk_ML = out.mirror.nk;
    layer_rough_ML = out.mirror.rough;
    
    layer_thick_ML = [zeros(1,size(layer_thick_ML,2));layer_thick_ML];
    layer_nk_ML = cat(2,ones(size(layer_nk_ML,1),1,size(layer_nk_ML,3)),layer_nk_ML);
    layer_rough_ML = [zeros(1,size(layer_rough_ML,2));layer_rough_ML];
    
    [r_ML,~,M_ML] = calcFresnel_rough_periodic(N_ML,size(layer_thick_ML,1) - n_per,layer_thick_ML,layer_nk_ML,layer_rough_ML,lambdaTheta,pol);
    r_ML = f(r_ML);
    correction_factor = sqrt(1 - correlated_roughness*0.2878);
    r_ML = r_ML.*correction_factor; 
    M_ML(2,1,:,:) = M_ML(2,1,:,:).*reshape(correction_factor,1,1,1,[]);
    
    layer_thick_abs = out.absorber.thick;
    layer_nk_abs = out.absorber.nk;
    layer_rough_abs = out.absorber.rough;
    
    [r_abs,t_abs,M_abs] = calcFresnel_rough_periodic(0,size(layer_thick_abs,1),layer_thick_abs,layer_nk_abs,layer_rough_abs,lambdaTheta,pol);
    r_abs = f(r_abs);
    t_abs = f(t_abs);
    layer_thick_etch = out.etch.thick;
    layer_nk_etch = out.etch.nk;
    layer_rough_etch = out.etch.rough;
    
    [r_etch,t_etch,M_etch] = calcFresnel_rough_periodic(0,size(layer_thick_etch,1),layer_thick_etch,layer_nk_etch,layer_rough_etch,lambdaTheta,pol);
    
    r_etch = f(r_etch);
    t_etch = f(t_etch);
    
    M_out = matMul2x2(M_abs,M_ML);
    M_MLabs = M_out;
    r_MLabs = vec( M_out(2,1,:) ./ M_out(1,1,:) );
    r_MLabs = scale_factor.*reshape(r_MLabs,size(lambdaTheta,1),[]);
    M_out = matMul2x2(M_etch,M_ML);
    M_MLetch = M_out;
    r_MLetch = vec( M_out(2,1,:) ./ M_out(1,1,:) );
    r_MLetch = scale_factor.*reshape(r_MLetch,size(lambdaTheta,1),[]);
end