using Statistics,DifferentialEquations
include("soluteperm.jl")
#Define analytical pressure function
function a_pressure(r,α)
    return r == 0 ? 1 - csch(α) : 1 - sinh(α*r)/(r*sinh(α))
    #return 1 - sinh(α*r)/(r*sinh(α))
end
#Define analytical pressure gradient function
function a_pressure_prime(r,α)
    return r == 0 ? 0 : (sinh(α*r)-α*r*cosh(r*α))/(sinh(α)*r^2)
    #return (sinh(α*r)-α*r*cosh(r*α))/(sinh(α)*r^2)
end
function MSTHighPeclet!(dc,c,p,t)
    #dc = preallocated vector corresponding to f(c,t)= dC/dt
    #c = concentration vector at all spatial points
    #p = vector of parameters as define in modelanalysis
    #t = current time
    #Unpackage parameters
    P,N,sigma,Perm,Lpt,Svt,Kt,Pvv,Pv,D,r,dr,kd=p
    co=1 #initial concentration
    tspan=3600 #timespan
    cv=co*exp(-t*tspan/kd) #time-dependent exponential decay term
    Pe=Lpt*Pv*(Pvv-P[1])*(1-sigma)/Perm #Initial Peclet value
    #interior boundary condition
    dc[1]=tspan*(2*D*(c[2]-c[1])/dr^2 +Lpt*Svt*Pv*(Pvv-P[1])*(1-sigma)*cv)
    for j=2:N-1
        dc[j]=tspan*(((2*D/r[j])*((c[j+1]-c[j])/dr))+(D*(c[j+1]-2*c[j]+c[j-1])/dr^2)+ Kt*((P[j+1]-P[j-1])/(2*dr))*((c[j+1]-c[j])/dr)+ Lpt*Svt*Pv*(Pvv-P[j])*(1-sigma)*cv)
    end
    dc[N]=0 # concentration at the outtermost boundary
end
function MST!(dc,c,p,t)
    #dc = preallocated vector corresponding to f(c,t)= dC/dt
    #c = concentration vector at all spatial points
    #p = vector of parameters as define in modelanalysis
    #t = current time
    #Unpackage parameters
    P,N,sigma,Perm,Lpt,Svt,Kt,Pvv,Pv,D,r,dr,kd=p
    co=1 #initial concentration
    tspan=3600 #timespan
    cv=co*exp(-t*tspan/kd) #time-dependent exponential decay term
    Pe=Lpt*Pv*(Pvv-P[1])*(1-sigma)/Perm #Initial Peclet value
    dc[1]= tspan*(2*D*(c[2]-c[1])*dr^-2 + Lpt*Svt*Pv*(Pvv-P[1])*(1-sigma)*(cv*exp(Pe)-c[1])/(exp(Pe)-1)) # R=0 boundary condition
    for j=2:N-1
        #Define interior nodes
        Pe=Lpt*Pv*(Pvv-P[j])*(1-sigma)/Perm #Peclet number
        dc[j]=tspan*(((2*D/r[j])*((c[j+1]-c[j])/dr))+(D*(c[j+1]-2*c[j]+c[j-1])/dr^2)+ Kt*((P[j+1]-P[j-1])/(2*dr))*((c[j+1]-c[j])/dr)+ Lpt*Svt*Pv*(Pvv-P[j])*(1-sigma)*(cv*exp(Pe)-c[j])/(exp(Pe)-1))
    end
    dc[N]=0 # concentration at the outtermost boundary
end
function Accumulation_Model(sol,n_spatial,n_time,n_nodes)
    c_model=zeros(n_nodes)
    spacingfactor = n_time ÷ n_nodes
    for j=1:n_nodes
        c_model[j]=mean(sol[:,spacingfactor*j])
    end
    return c_model
end
function accumlation_intermediate_rhod(x) #x is a vector containing Lpt and Kt
    n_spatial=100
    n_time=21
    n_nodes = 21 # number of nodes desired to be outputted for accumulation data
    Lpt,Kt = x
    Svt = 200;      #tumor vascular density
    D = 2e-07;  #solute diffusion coefficient
    rs = 13/2;      #partical radius (nm)
    Perm,sigma=soluteperm(Lpt,rs) #Vascular permeability and solute reflection coefficient respectively
    R=1. # Tumor Radius
    Pv=25. #
    Pvv=1.
    kd=1480*60 # blood circulation time of drug in hours;
    att=R*sqrt(Lpt*Svt/Kt) #parameter alpha for tumor, ignoring lymphatics
    Press(r)= a_pressure(r,att) # Define pressure function for specific Lpt,Kt
    co=1.
    tdomain=300 #Define time domain
    #Define spatial domain
    r= (1/R)*(range(0,stop=R,length=n_spatial))
    dr=1/(n_spatial-1)

    P= broadcast(Press,r) # Calculate pressure profile

    c0=zeros(n_spatial,1)
    c0[end]=0. #Initial concentration at outside boundary
    #define timespan
    time_end =(tdomain/3600)
    tspan=(0.,time_end)
    #package parameters
    p=P,n_spatial,sigma,Perm,Lpt,Svt,Kt,Pvv,Pv,D,r,dr,kd
    #Define and solve system of ODEs using time constant of 1/n_nodes and the QNDF solver for stiff systems
    dt=time_end/(n_time-1)
    #t = dt*[i for i=0:n_nodes-1]
    prob=ODEProblem(MST!,c0,tspan,p)
    sol= DifferentialEquations.solve(prob,QNDF(),saveat=dt)
    if length(sol.t)==1
        prob=ODEProblem(MSTHighPeclet!,c0,tspan,p)
        sol= DifferentialEquations.solve(prob,QNDF(),saveat=dt)
    end
    accum_dat = Accumulation_Model(sol,n_spatial,n_time,n_nodes)
    return accum_dat
end
function accumlation_intermediate_FITC(x) #x is a vector containing Lpt and Kt
    n_spatial=100
    n_time=21
    n_nodes = 21 # number of nodes desired to be outputted for accumulation data
    Lpt,Kt = x
    Svt = 200;      #tumor vascular density
    D = 1.375e-07;  #solute diffusion coefficient
    rs = 16;      #partical radius (nm)
    Perm,sigma=soluteperm(Lpt,rs) #Vascular permeability and solute reflection coefficient respectively
    R=1. # Tumor Radius
    Pv=25. #
    Pvv=1.
    kd=1278*60 # blood circulation time of drug in hours;
    att=R*sqrt(Lpt*Svt/Kt) #parameter alpha for tumor, ignoring lymphatics
    Press(r)= a_pressure(r,att)
    co=1.
    tdomain=300

    r= (1/R)*(range(0,stop=R,length=n_spatial))
    dr=1/(n_spatial-1)

    P= broadcast(Press,r) # Calculate pressure profile

    c0=zeros(n_spatial,1)
    c0[end]=0.
    #define timespan
    time_end =(tdomain/3600)
    tspan=(0.,time_end)
    #package parameters
    p=P,n_spatial,sigma,Perm,Lpt,Svt,Kt,Pvv,Pv,D,r,dr,kd
    #Define and solve system of ODEs using time constant of 1/n_nodes and the QNDF solver for stiff systems
    dt=time_end/(n_time-1)
    #t = dt*[i for i=0:n_nodes-1]
    prob=ODEProblem(MST!,c0,tspan,p)
    sol= DifferentialEquations.solve(prob,QNDF(),saveat=dt)
    if length(sol.t)==1
        prob=ODEProblem(MSTHighPeclet!,c0,tspan,p)
        sol= DifferentialEquations.solve(prob,QNDF(),saveat=dt)
    end
    accum_dat = Accumulation_Model(sol,n_spatial,n_time,n_nodes)
    return accum_dat
end
function Isolated_Pressure(N,R,Lpt,Svt,Kt,Pvv)

    # Pressure Profile
    att = R*sqrt(Lpt*Svt/Kt)

    r = range(0,step = R/(N-1),length = N)
    r = r/R
    dr = 1.0/(N-1)
    ivdr2 = 1.0./dr^2
    att2 = att^2
    A = zeros(typeof(Lpt),N,N)
    F = zeros(typeof(Lpt),N,1)

    M = N
    for i in 2:M-1
        A[i,i-1] = -1.0/r[i]/dr + ivdr2
        A[i,i] = -2.0*ivdr2 - (att2)
        A[i,i+1] = 1.0/r[i]/dr + ivdr2
        F[i] = -(att2)*Pvv
    end

    # Boundary condtions for isolated tumor model
    A[1,1] = -2.0/dr^2 - (att^2)
    A[1,2] = 2.0/dr^2
    F[1] = -att2*Pvv
    A[N,N] = 1.0

    # Solution of linear problem for pressure distribution
    P = A\F
    return P
end
