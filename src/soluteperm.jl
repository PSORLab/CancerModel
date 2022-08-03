function sumseries(term,start,stop)
    sum = 0
    for i=start:stop
        sum += term(i)
    end
    return sum
end

function soluteperm(Lpt,rs)
    #calculate diffusion constant from Stoke's Einstein equation
    kB = 1.380648*10^(-23)       # Boltzmann Constant (J/K)
    T = 310.15                   # Temperature K
    eta = 3*10^(-5)              # viscosity of blood (mmHg-sec)
    conv_mu  = 133.322365        # (Pascal/mmHg)
    etac = eta*conv_mu            # Pascal-sec
    pore_conv = 10^(-9)          #(m/nm)
    r_partc = rs*pore_conv       #radius (m)
    D0 = kB*T/(6*pi*etac*r_partc)*1e4 # Diffusivity (cm^2/s)

    #Bungay and Brenne Hydrodynamic coefficients for cylindrical pore modal
    a_k= [-73/60,77293/50400,-22.5083,-5.6117,-.3363,-1.216,1.647]
    b_k= [7/60,-2227/50400,4.0180,-3.9788,-1.9215,4.392,5.006]

    #Additional constants
    gamma = 1e-3 #fraction of vascular surface area occupie by pores
    eta = 3e-5  #Blood viscosity (mmHg/s)
    L = 5e-4    # Vessel wall thickness (cm)
    r_pore = sqrt(8*eta*L*Lpt/gamma)*1e7 # vessel radius of pore in nm
    lambda = rs/r_pore # ratio of diffusing marcomolecule radius to pore radisu

    #Calculate ratio of K_t & K_s factors
    prefactor= (9/4)*pi^2*(sqrt(2))*((1-lambda)^-2.5) # use for calculation of kt &ks
    K_t = prefactor*(1+sumseries((i->a_k[i]*(1-lambda)^i),1,2))+ sumseries(i->(a_k[i+3])*lambda^i,0,4)
    K_s= prefactor*(1+sumseries((i->b_k[i]*(1-lambda)^i),1,2))+ sumseries(i->(b_k[i+3])*lambda^i,0,4)

    #calculate remaining constants
    Phi = (1-lambda)^2 #partition coefficient
    H=6*pi*Phi/K_t #diffusive hindrance factor
    W= Phi*(2-Phi)*K_s/(2*K_t) #convective hindrance factor
    Perm= gamma*H*D0/L #Vascular permeability
    sigma=1-W #solute reflection coefficient
    return Perm, sigma
end
