# Import Packages and External Functions
using ValidatedNumerics,
    DelimitedFiles, LinearAlgebra, Statistics, BenchmarkTools
using JuMP, ForwardDiff, Ipopt, EAGO
using Flux
using BSON: @load
using CSV, DataFrames, Plots
include("surrogate_optimization_functions.jl")

#Define Local Activation Function
swish_local(x) = x / (1 + exp(-x))

#Define Feed Forward Neural Network
function NN_explicit(x, W)
    #Initialize Weights & Biases
    w1, w2, w3, w4, b1, b2, b3, b4 = W
    #Feed Forward
    l1 = (w1 * x + b1)
    l2 = swish_local.(w2 * l1 + b2)
    l3 = swish_local.(w3 * l2 + b3)
    #l2=EAGO.swish1.(w2*l1+b2)
    #l3=EAGO.swish1.(w3*l2+b3)
    l4 = (w4 * l3 + b4)
    return l4
end
# Define function for generating empirical accumulation data
function obtain_data(idx, n_time, drug)
    #idx - 1 = control case, 2 = 3mg case, 3= 30mg case
    #n_time = Number of time nodes
    #drug = Either "rhod" or "FITC"
    Svt = 200.0 #Vascular Density
    Peff =
        drug == "rhod" ? [9.59873e-07, 4.6098e-06, 2.80047e-06][idx] :
        [8.18378e-07, 4.30307e-06, 1.62231e-06][idx] # Set Effective Permeability
    kd = drug == "rhod" ? 1480.0 * 60 : 1278.0 * 60 #Set Circulation Half Life
    #Specify Time Horizon
    t0 = 0.0
    tf = 300.0
    c0 = 1.0 # Initial concentration
    t_data = zeros(Float64, n_time)
    c_data = zeros(Float64, n_time)
    for i = 1:n_time
        t_data[i] = t0 + ((tf - t0) / (n_time - 1)) * (i - 1)
        c_data[i] =
            (c0 * Peff * Svt * kd / (1 - Peff * Svt * kd)) *
            (exp(-Peff * Svt * t_data[i]) - exp(-t_data[i] / kd))
    end
    return t_data, c_data
end
# Define Local Optimizer using Ipopt - mostly used for testing before running global optimization
function Local_Optim(p_lo, p_hi, c_data, surrogate, idx)
    #p_lo,p_hi = lower and upper bound on search space for Lpt and Kt
    #c_data= empirical data to fit model to
    #Surrogate = ANN representing mechanistic model
    #idx - 1 = control case, 2 = 3mg case, 3= 30mg case
    #Specify Problem
    model = Model(with_optimizer(Ipopt.Optimizer, max_iter = 10000))
    obj(Lpt, Kt) =
        dot(surrogate([Lpt, Kt]) - c_data, surrogate([Lpt, Kt]) - c_data) #Sum of Squared Errors objective function
    JuMP.register(model, :obj, 2, obj, autodiff = true) #register objective function
    #JuMP.register(model,:ip_constr,2,ip_constr,autodiff=true) #register empirical superficial pressure constraint
    #Specify program variables
    @variable(model, p_lo[1] <= Lpt <= p_hi[1])
    @variable(model, p_lo[2] <= Kt <= p_hi[2])

    #Specify lower and upper bound for superifical pressure constraint using affine approximations
    m_l= [0.285453383252671, 0.7354533734422002, 1.589830508474576]
    m_u= [0.39665271966527194, 1.0577489687684147, 2.444710211591536]

    JuMP.@NLconstraint(model, pcon_ub, Kt <= m_u[idx] * Lpt)
    JuMP.@NLconstraint(model, pcon_lb, Kt >= m_l[idx] * Lpt)
    #=
    ifp_exp=[5.27,3.32,2.2]
    ifp_CE=[0.4,0.3,0.25]
    JuMP.@NLconstraint(model,pcon_lb,ip_constr(Lpt,Kt)>=ifp_exp[idx]-conf_int[idx])
    JuMP.@NLconstraint(model,pcon_ub,ip_constr(Lpt,Kt)<=ifp_exp[idx]+conf_int[idx])
    =#
    #Solve problem and return values
    @NLobjective(model, Min, obj(Lpt, Kt))
    JuMP.optimize!(model)
    fval = JuMP.objective_value(model)
    psol = (JuMP.value.(Lpt), JuMP.value.(Kt))
    return psol, fval
end

function Global_Optim(p_lo, p_hi, c_data, surrogate, idx)
    #p_lo,p_hi = lower and upper bound on search space for Lpt and Kt
    #c_data= empirical data to fit model to
    #Surrogate = ANN representing mechanistic model
    #idx - 1 = control case, 2 = 3mg case, 3= 30mg case
    #Specify Problem
    atol = idx == 1 ? 1e-9 : 1e-7 #set absolute tolerance based on problem
    model = Model(
        with_optimizer(
            EAGO.Optimizer,
            absolute_tolerance = atol,
            relative_tolerance = 1e-1,
            iteration_limit = Int(1E6),
            verbosity = 1,
            time_limit = 100000.0,
        ),
    )

    obj(Lpt, Kt) =
        dot(surrogate([Lpt, Kt]) - c_data, surrogate([Lpt, Kt]) - c_data) #Sum of Squared Errors objective function
    JuMP.register(model, :obj, 2, obj, autodiff = true)#register objective function
    #JuMP.register(model,:ip_constr,2,ip_constr,autodiff=true)
    @variable(model, p_lo[1] <= Lpt <= p_hi[1])
    @variable(model, p_lo[2] <= Kt <= p_hi[2])
    #Specify lower and upper bound for superifical pressure constraint using affine approximations
    #Specify lower and upper bound for superifical pressure constraint using affine approximations
    m_l= [0.285453383252671, 0.7354533734422002, 1.589830508474576]
    m_u= [0.39665271966527194, 1.0577489687684147, 2.444710211591536]

    JuMP.@NLconstraint(model, pcon_ub, Kt <= m_u[idx] * Lpt)
    JuMP.@NLconstraint(model, pcon_lb, Kt >= m_l[idx] * Lpt)
    #Solve problem and return values
    @NLobjective(model, Min, obj(Lpt, Kt))
    JuMP.optimize!(model)
    fval = JuMP.objective_value(model)
    psol = (JuMP.value.(Lpt), JuMP.value.(Kt))
    return psol, fval
end

# Define Routine For Initializing surrogate based on ANN for each case and call global optimization program
function Global_Optim_rout(NN, idx, lb, ub, p_lo, p_hi, drug)
    #NN = ANN representing mechanistic model
    #idx - 1 = control case, 2 = 3mg case, 3= 30mg case
    #lb,ub = lower and upper bound from ANN training data
    #p_lo,p_hi = lower and upper bound for optimization program
    #drug = Either "rhod" or "FITC"

    #Define scalers for ANN Model
    scaler(x) = (x - lb) ./ (ub - lb) #
    invscaler(x_sc) =[x_sc[1] * (ub[1] - lb[1]) + lb[1], x_sc[2] * (ub[2] - lb[2]) + lb[2]]
    output_scaler(x) = x ./ 0.3
    inv_output_scaler(x) = x .* 0.3

    #Decompose Flux model and extract weights and biases to initialize locally defined feed forward neural network
    wb, re = Flux.destructure(NN)
    w1 = reshape(wb[1:48], (24, 2))
    b1 = reshape(wb[49:72], (24, 1))
    w2 = reshape(wb[73:648], (24, 24))
    b2 = reshape(wb[649:672], (24, 1))
    w3 = reshape(wb[673:1248], (24, 24))
    b3 = reshape(wb[1249:1272], (24, 1))
    w4 = reshape(wb[1273:1776], (21, 24))
    b4 = reshape(wb[1777:1797], (21, 1))
    W = w1, w2, w3, w4, b1, b2, b3, b4

    #Set up scaled ANN surrogate model
    surrogate(p) = inv_output_scaler(NN_explicit(scaler(p), W))

    #Generate Empirical Data and call optimization program
    n_time = 21
    t_data, c_data = obtain_data(idx, n_time, drug)
    psol, fval = Global_Optim(p_lo, p_hi, c_data, surrogate, idx)

    #Display Results
    @show idx
    @show Lpt, Kt = psol
    @show SSE = fval
    modeldat = (drug == "rhod" ? accumlation_intermediate_rhod([Lpt, Kt]) : accumlation_intermediate_FITC([Lpt, Kt])) #Generate mechanistic model data
    #Plot Results
    optplot = plot(
        t_data,
        [c_data, surrogate([Lpt, Kt]), modeldat],
        title = "$(uppercasefirst(drug)) Accumulation- Data/Surrogate/Model-$idx",
        ylabel = "Accmulation",
        xlabel = "Time (seconds)",
        label = ["Data" "Surrogate" "Model"],
        legend = :topleft,
    )
    display(optplot)
    #savefig(optplot, "P:\\FinalModels\\$(drug)globaloptaccumplot$idx.png") #optionally save figure
    return Lpt, Kt
end
#Return Pore Size
function poresize(Lpt)
    gamma = 1e-3
    eta = 3e-5
    L = 5e-6
    return sqrt(8 * eta * L * Lpt * 0.01 / gamma) * 1e9 * 2
end
#=
ip_constr1(Lpt,Kt)= 25*(1 - sinh(sqrt(Lpt*200/Kt)*(98/99))/((98/99)*sinh(sqrt(Lpt*200/Kt))))
function ip_constr(Lpt,Kt)
    N=100
    R=1.
    Svt=200.
    Pvv=1.
    P_profile = Isolated_Pressure(N,R,Lpt,Svt,Kt,Pvv);
    IFP = P_profile[99]
    return 25*IFP
end
=#
#Calculate Mean Interstitial Fluid Pressure
function interstit_pressure(Lpt, Kt)
    Svt = 200.0
    α = sqrt(Lpt * Svt / Kt)
    r = 0.0:(1/99):1
    press = broadcast(x -> a_pressure(x, α), r)
    return 25 * mean(press)
end
#Output Routine for calling and storing calculations of Lpt,Kt,Pore Size, and Mean IFP
function output(NN, idx, lb, ub, p_lo, p_hi, drug)
    Lpt, Kt = Global_Optim_rout(NN, idx, lb, ub, p_lo, p_hi, drug)
    pore_size = poresize(Lpt)
    IFP = interstit_pressure(Lpt, Kt)
    return [Lpt, Kt, pore_size, IFP]
end


############################################################################
#Load in 3mg/30mg Neural Networks and Define Bounds
@load "P:\\FinalModels\\rhodaminelowhigh.bson" NN
NN_dosage_rhod = NN
@load "P:\\FinalModels\\FITClowhigh.bson" NN
NN_dosage_FITC = NN
dosage_lb = [5e-7, 7e-7]
dosage_ub = [3.5e-6, 4e-6]

#Load in Control Case Neural Networks and Define Bounds
@load "P:\\FinalModels\\rhodaminecontrol.bson" NN
NN_control_rhod = NN
@load "P:\\FinalModels\\FITCcontrol.bson" NN
NN_control_FITC = NN
control_lb = [1e-7, 1e-7]
control_ub = [1.75e-6, 1e-6]

#Optionally Restrain Search Space
#=
p_lo1,p_hi1=[8.5e-7,3.3e-7],[8.6e-7,3.4e-7]
p_lo2,p_hi2=[2.75e-6,2e-6],[2.85e-6,2.1e-6]
p_lo3,p_hi3=[1.1e-6,1.75e-6],[1.2e-6,1.85e-6]
=#

#Call output function and store results in dataframe
df = DataFrame()
@show df.control_rhod = output(NN_control_rhod,1,control_lb,control_ub,control_lb,control_ub,"rhod")
@show df.threemg_rhod = output(NN_dosage_rhod,2,dosage_lb,dosage_ub,dosage_lb,dosage_ub,"rhod")
@show df.thirtymg_rhod = output(NN_dosage_rhod,3,dosage_lb,dosage_ub,dosage_lb,dosage_ub,"rhod")

df.space = [" ", " ", " ", " "]

@show df.control_FITC = output(NN_control_FITC,1,control_lb,control_ub,control_lb,control_ub,"FITC")
@show df.threemg_FITC = output(NN_dosage_FITC,2,dosage_lb,dosage_ub,dosage_lb,dosage_ub,"FITC")
@show df.thirtymg_FITC = output(NN_dosage_FITC,3,dosage_lb,dosage_ub,dosage_lb,dosage_ub,"FITC")

#Write Data Frame to CSV
CSV.write("P:\\FinalModels\\masterglobalreadouts_test.csv", df)

println("Done")
