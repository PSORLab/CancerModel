using ValidatedNumerics
using IntervalArithmetic
using DelimitedFiles
using LinearAlgebra
using Statistics
using BenchmarkTools
using CSV
using JuMP, ForwardDiff, Ipopt
using EAGO
using Flux
using BSON: @load

## Optimization
function Local_Optim(p_lo,p_hi)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol",1e-6)
    set_optimizer_attribute(model, "bound_frac",0.5*rand(1)[1])
    #model = Model(with_optimizer(Ipopt.Optimizer,tol = 0.001 ,bound_frac = 0.5*rand(1)[1]))
    obj(dose,rs) = -NNm([dose,rs])
    JuMP.register(model,:obj,2,obj,autodiff=true)
    noncon_s(dose,rs) = NNp([dose,rs])-4.5
    JuMP.register(model,:noncon_s,2,noncon_s,autodiff=true)
    noncon_p(dose,rs) = NNp([dose,rs])-3.6
    JuMP.register(model,:noncon_p,2,noncon_p,autodiff=true)
    @variable(model, p_lo[1] <= dose <= p_hi[1])
    @variable(model, p_lo[2] <= rs <= p_hi[2])
    @NLobjective(model, Min, obj(dose,rs))
    @NLconstraint(model, noncon_s(dose,rs) <= 0.0)
    @NLconstraint(model, noncon_p(dose,rs) >= 0.0)

    JuMP.optimize!(model)
    @show fval = JuMP.objective_value(model)
    psol = (JuMP.value.(dose), JuMP.value.(rs))
    return psol
end
function Global_Optim(p_lo,p_hi)
    model = Model(
        with_optimizer(
            EAGO.Optimizer,
            absolute_tolerance = 1e-7,
            relative_tolerance = 1e-1,
            iteration_limit = Int(1E6),
            verbosity = 1,
            time_limit = 100000.0,
            log_on=true
        ),
    )
    obj(dose,rs) = -NNm([dose,rs])
    JuMP.register(model,:obj,2,obj,autodiff=true)
    noncon_s(dose,rs) = NNp([dose,rs])-4.5
    JuMP.register(model,:noncon_s,2,noncon_s,autodiff=true)
    noncon_p(dose,rs) = NNp([dose,rs])-3.6
    JuMP.register(model,:noncon_p,2,noncon_p,autodiff=true)
    @variable(model, p_lo[1] <= dose <= p_hi[1])
    @variable(model, p_lo[2] <= rs <= p_hi[2])
    @NLobjective(model, Min, obj(dose,rs))
    @NLconstraint(model, noncon_s(dose,rs) <= 0.0)
    @NLconstraint(model, noncon_p(dose,rs) >= 0.0)

    JuMP.optimize!(model)
    @show fval = JuMP.objective_value(model)
    psol = (JuMP.value.(dose), JuMP.value.(rs))
    return psol
end


#---------------------------------------
#Initialize ANN Models
function NN_explicit(x, W)
    #Initialize Weights & Biases
    w1, w2, w3, w4, b1, b2, b3, b4 = W
    #Feed Forward
    l1 = (w1 * x + b1)
    l2 = tanh.(w2 * l1 + b2)
    l3 = tanh.(w3 * l2 + b3)
    l4 = (w4 * l3 .+ b4)
    return l4[1]
end

lb=Float32[0.0,5.0]
ub=Float32[30.0,10.0]
#Define scaler functions to preprocess data
vecscaler(x)=(x.-lb)./(ub.-lb)
scaler(x)= ((x[1]- lb[1])/(ub[1]-lb[1]),(x[2]- lb[2])/(ub[2]-lb[2]))
invscaler(x_sc)=(x_sc[1]*(ub[1]-lb[1]) + lb[1],x_sc[2]*(ub[2]-lb[2]) + lb[2])
output_scaler(x)= x ./ .3
inv_output_scaler(x)=x.*.3
con_output_scaler(x)= x ./ 2
con_inv_output_scaler(x)=x.*2
@load "P:\\TherapyDrugDesign\\cmean.bson" NN_mean
wb_mean, re_mean = Flux.destructure(NN_mean)
w1m = reshape(wb_mean[1:16], (8, 2))
b1m = reshape(wb_mean[17:24], (8, 1))
w2m = reshape(wb_mean[25:88], (8, 8))
b2m = reshape(wb_mean[89:96], (8, 1))
w3m = reshape(wb_mean[97:160], (8, 8))
b3m = reshape(wb_mean[161:168], (8, 1))
w4m = reshape(wb_mean[169:176], (1, 8))
b4m = wb_mean[177]
W_mean = w1m, w2m, w3m, w4m, b1m, b2m, b3m, b4m
@load "P:\\TherapyDrugDesign\\cpeak.bson" NN_peak
wb_peak, re_peak = Flux.destructure(NN_peak)
w1p = reshape(wb_peak[1:16], (8, 2))
b1p = reshape(wb_peak[17:24], (8, 1))
w2p = reshape(wb_peak[25:88], (8, 8))
b2p = reshape(wb_peak[89:96], (8, 1))
w3p = reshape(wb_peak[97:160], (8, 8))
b3p = reshape(wb_peak[161:168], (8, 1))
w4p = reshape(wb_peak[169:176], (1, 8))
b4p = wb_peak[177]
W_peak = w1p, w2p, w3p, w4p, b1p, b2p, b3p, b4p
NNm(x)= inv_output_scaler(NN_explicit(vecscaler(x),W_mean))
NNp(x)= con_inv_output_scaler(NN_explicit(vecscaler(x),W_peak))


#------------------------------------------------------------------
#Call Optimization Routines
p_lo = [0.0,5.0]
p_hi = [30.0,10.0]
@show psol = Global_Optim(p_lo,p_hi)
obj(dose,rs) = -NNm([dose,rs])
#@show obj(4.3567,6.288)
noncon_s(dose,rs) = NNp([dose,rs])-4.5
noncon_p(dose,rs) = NNp([dose,rs])-3.6
#@show noncon_s(4.3567,6.288)
#@show noncon_p(4.3567,6.288)
