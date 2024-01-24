using ModelingToolkit,MethodOfLines,DifferentialEquations,DomainSets,IfElse,Statistics
using Symbolics: @register_symbolic

@parameters t r
@parameters K,L_p,S_V,p_v,R,D,c_o,rs, k_d,α,σ,P
@variables c(..)

Dt = Differential(t)
Dr = Differential(r)
Drr = Differential(r)^2

function p(r,α)
    return IfElse.ifelse(r == 0,1. - csch(α),1. - sinh(α*r)/(r*sinh(α)))
end
@register_symbolic p(r,α)
function dpdr(r,α)
    return IfElse.ifelse(r == 0,0.0,(sinh(α*r)-α*r*cosh(r*α))/(sinh(α)*r^2))
end
@register_symbolic dpdr(r,α)

r_min = t_min =0.0;
r_max = 1.0;
t_max = (300/3600);


eq=Dt(c(t,r)) ~ 3600*((2*D/r)*Dr(c(t,r))+D*Drr(c(t,r))+K*dpdr(r,α)*Dr(c(t,r))+L_p*S_V*p_v*(1. - p(r,α))*(1 - σ)*c_o*exp(-3600*t/k_d) + P*S_V*(c_o*exp(-3600*t/k_d) - c(t,r))*((L_p*p_v*(1. -p(r,α))*(1-σ)/P)/exp(L_p*p_v*(1. -p(r,α))*(1-σ)/P)-1))
domains = [r ∈ Interval(r_min,r_max),t ∈ Interval(t_min,t_max)]

bcs = [
    Dr(c(t,r_min)) ~ 0.0,
    c(t,r_max) ~ 0.0,
    c(0,r) ~ 0.0
]

function Diffusion(rs)
    Diam_parc = 2*rs
    return D = 1.981e-06*Diam_parc^(-1.157) + 2.221e-08
end

function Blood_half_life(rs)
    Diam_parc = 2.0*rs
    a1 = 1081
    b1 = -16.63
    c1 = 84.82
    a2 = 517.4
    b2 = 65.61
    c2 = 996.6
    return kd = (a1*exp(-((Diam_parc-b1)/c1)^2) + a2*exp(-((Diam_parc-b2)/c2)^2))*60
end

function solutePerm(Lpt,rs)
    # calculate diffusion coefficient from Stoke's Einstein
    kB = 1.380648*10.0^(-23)               # Boltzmann Constant (J/K)
    Temp = 310.15                           # Temperature K
    eta = 3*10.0^(-5)                      # viscosity of blood (mmHg-sec)
    conv_mu  = 133.322365                # (Pascal/mmHg)
    etac = eta*conv_mu                   # Pascal-sec
    pore_conv = 10.0^(-9)                  # (m/nm)
    r_partc = rs*pore_conv               # radius (m)
    D0 = kB*Temp/(6*pi*etac*r_partc)*1.0e4;   # Diffusivity (cm^2/s)

    # Bungay and Brenner
    a = [-73/60,77293/50400,-22.5083,-5.6117,-0.3363,-1.216,1.647]
    b = [7/60;-2227/50400;4.0180;-3.9788;-1.9215;4.392;5.006]

    # Calculate the pore size
    gamma = 1.0e-3
    eta = 3.0e-5                              # Blood viscosity (mmHg/s)
    L = 5.0e-4                                # Vessel wall thickness (cm)
    r_pore::typeof(rs) = sqrt(8*eta*L*Lpt/gamma)*1.0e7    # nm
    # rs = 30;                              # solute radius (nm) 30nm for FITC
    lambda = rs/r_pore
    t1 = zero(typeof(rs))
    t2 = zero(typeof(rs))
    p1 = zero(typeof(rs))
    p2 = zero(typeof(rs))
    for i = 1:7
        if i<3
            t1 = t1 + a[i]*(1-lambda)^i
            p1 = p1 + b[i]*(1-lambda)^i
        else
            t2 = t2 + a[i]*lambda^(i-3)
            p2 = p2 + b[i]*lambda^(i-3)
        end
    end
    Kt = t2 + 9.0/4.0*pi^2*sqrt(2.0)*(1.0+t1)*sqrt(1.0-lambda)^(-5)
    Ks = p2 + 9.0/4.0*pi^2*sqrt(2.0)*(1.0+p1)*sqrt(1.0-lambda)^(-5)
    Phi = (1.0-lambda)^2
    H = 6.0*pi*Phi/Kt
    W = Phi*(2.0-Phi)*Ks/(2.0*Kt)
    Perm = gamma*H*D0/L
    sigma = 1.0 - W
    return Perm,sigma
end

function getparam(Lpt,Kt,rs_v)
    rs_v=16.
    Perm,sigma = solutePerm(Lpt,rs_v) #Permeability and solute reflection coefficient
    #bhl = Blood_half_life(rs_v) #Kd
    bhl = 1278.0*60.0;
    #D_coeff = Diffusion(rs_v)
    D_coeff=1.375e-07;
    S_V_val=200.
    p_v_val=25.
    R_val=1.
    c_o_val = 1.
    alpha = R_val*sqrt(Lpt*S_V_val/Kt)

    @parameters K,L_p,S_V,p_v,R,D,c_o,p_ss,rs, k_d,α,σ,P
    param = [K=>Kt,L_p=>Lpt,S_V=>S_V_val,p_v=>p_v_val,R=>R_val,D=>D_coeff,c_o=>c_o_val,rs=>rs_v,k_d=>bhl,α=>alpha,σ=>sigma,P=>Perm]
    return param
end
param = getparam(1e-6,1e-6,16.)
@named pdesys = PDESystem(eq, bcs, domains, [t, r], [c(t, r)],param)

N = 101.
dr = (r_max-r_min)/N
order = 2
discretization = MOLFiniteDifference([r=>dr], t, approx_order=order, grid_align=center_align)

prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
n_time =21
dt=t_max/(n_time-1)





sol = solve(prob,QNDF(),saveat=dt)
function Accumulation_Model(sol,n_spatial,n_time,n_nodes)
    c_model=zeros(n_nodes)
    spacingfactor = n_time ÷ n_nodes
    for j=1:n_nodes
        c_model[j]=mean(sol[:,spacingfactor*j])
    end
    return c_model
end

grid = get_discrete(pdesys, discretization)
discrete_r = grid[r]
discrete_t = sol[t]
Accumulation_Model(sol[c(t,r)],101,21,21)


using Plots
anim = @animate for i in 1:length(sol.t)
    p1 = plot(discrete_r, map(d -> sol[d][i], grid[c(t, r)]), label="c, t=$(discrete_t[i])[1:9] "; legend=false, xlabel="r",ylabel="c")
    plot(p1)
end
gif(anim, "plot.gif",fps=5)
