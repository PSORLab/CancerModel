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
    obj(lpt,kt,rs) = -NNm([lpt,kt,rs])
    JuMP.register(model,:obj,3,obj,autodiff=true)
    noncon_s(lpt,kt,rs) = NNp([lpt,kt,rs])-4.5
    JuMP.register(model,:noncon_s,3,noncon_s,autodiff=true)
    noncon_p(lpt,kt,rs) = NNp([lpt,kt,rs])-3.6
    JuMP.register(model,:noncon_p,3,noncon_p,autodiff=true)
    @variable(model, p_lo[1] <= lpt <= p_hi[1])
    @variable(model, p_lo[2] <= kt <= p_hi[2])
    @variable(model, p_lo[3] <= rs <= p_hi[3])
    @NLobjective(model, Min, obj(lpt,kt,rs))
    @NLconstraint(model, noncon_s(lpt,kt,rs) <= 0.0)
    @NLconstraint(model, noncon_p(lpt,kt,rs) >= 0.0)

    JuMP.optimize!(model)
    @show fval = JuMP.objective_value(model)
    psol = (JuMP.value.(lpt),JuMP.value.(kt), JuMP.value.(rs))
    return psol
end
function Global_Optim(p_lo,p_hi)
    model = Model(
        with_optimizer(
            EAGO.Optimizer,
            absolute_tolerance = 1e-5,
            relative_tolerance = 1e-1,
            iteration_limit = Int(1E6),
            verbosity = 1,
            time_limit = 100000.0,
            log_on=true
        ),
    )
    obj(lpt,kt,rs) = -NNm([lpt,kt,rs])
    JuMP.register(model,:obj,3,obj,autodiff=true)
    noncon_s(lpt,kt,rs) = NNp([lpt,kt,rs])-4.5
    JuMP.register(model,:noncon_s,3,noncon_s,autodiff=true)
    noncon_p(lpt,kt,rs) = NNp([lpt,kt,rs])-3.6
    JuMP.register(model,:noncon_p,3,noncon_p,autodiff=true)
    @variable(model, p_lo[1] <= lpt <= p_hi[1])
    @variable(model, p_lo[2] <= kt <= p_hi[2])
    @variable(model, p_lo[3] <= rs <= p_hi[3])
    @NLobjective(model, Min, obj(lpt,kt,rs))
    @NLconstraint(model, noncon_s(lpt,kt,rs) <= 0.0)
    @NLconstraint(model, noncon_p(lpt,kt,rs) >= 0.0)

    JuMP.optimize!(model)
    @show fval = JuMP.objective_value(model)
    psol = (JuMP.value.(lpt),JuMP.value.(kt), JuMP.value.(rs))
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

lb=Float32[5e-7,5e-7,5]
ub=Float32[5e-6,5e-6,30]
#Define scaler functions to preprocess data
vecscaler(x)=(x.-lb)./(ub.-lb)
scaler(x)= ((x[1]- lb[1])/(ub[1]-lb[1]),(x[2]- lb[2])/(ub[2]-lb[2]))
invscaler(x_sc)=(x_sc[1]*(ub[1]-lb[1]) + lb[1],x_sc[2]*(ub[2]-lb[2]) + lb[2])
output_scaler(x)= x ./ .6
inv_output_scaler(x)=x.*.6
con_output_scaler(x)= x ./ 2
con_inv_output_scaler(x)=x.*2
@load "P:\\TherapyDrugDesign\\cmeanEE3param.bson" NN_mean
wb_mean, re_mean =Flux.destructure(NN_mean)
w1m = reshape(wb_mean[1:36], (12, 3))
b1m = reshape(wb_mean[37:48], (12, 1))
w2m = reshape(wb_mean[49:192], (12, 12))
b2m = reshape(wb_mean[193:204], (12, 1))
w3m = reshape(wb_mean[205:348], (12, 12))
b3m = reshape(wb_mean[349:360], (12, 1))
w4m = reshape(wb_mean[361:372], (1, 12))
b4m = wb_mean[373]
W_mean = w1m, w2m, w3m, w4m, b1m, b2m, b3m, b4m
@load "P:\\TherapyDrugDesign\\cpeakEE3param.bson" NN_peak
wb_peak, re_peak = Flux.destructure(NN_peak)
w1p = reshape(wb_peak[1:36], (12, 3))
b1p = reshape(wb_peak[37:48], (12, 1))
w2p = reshape(wb_peak[49:192], (12, 12))
b2p = reshape(wb_peak[193:204], (12, 1))
w3p = reshape(wb_peak[205:348], (12, 12))
b3p = reshape(wb_peak[349:360], (12, 1))
w4p = reshape(wb_peak[361:372], (1, 12))
b4p = wb_mean[373]
W_peak = w1p, w2p, w3p, w4p, b1p, b2p, b3p, b4p
NNm(x)= inv_output_scaler(NN_explicit(vecscaler(x),W_mean))
NNp(x)= con_inv_output_scaler(NN_explicit(vecscaler(x),W_peak))


#------------------------------------------------------------------
#Call Optimization Routines
p_lo = [5e-7,5e-7,5]
p_hi = [5e-6,5e-6,30]
@show psol = Local_Optim(p_lo,p_hi)
obj(lpt,kt,rs) = -NNm([lpt,kt,rs])
#@show obj(4.3567,6.288)
noncon_s(lpt,kt,rs) = NNp([lpt,kt,rs])-4.5
noncon_p(lpt,kt,rs) = NNp([lpt,kt,rs])-3.6
#@show noncon_s(4.3567,6.288)
#@show noncon_p(4.3567,6.288)
