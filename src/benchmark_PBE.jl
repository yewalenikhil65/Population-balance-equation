# doing validation with analytical expression
using ModelingToolkit, LinearAlgebra
using DiffEqBase ,DiffEqJump, OrdinaryDiffEq
using LoopVectorization, Plots
using BenchmarkTools
using SpecialFunctions, Statistics
plotly()
## Parameter
N = 100;      # Number of clusters
C  = 238.720;  # initial concentration == number of initial monomers/ volume of the bulk system
uₒ = 10000; # No. of singlets initially
V = uₒ/C;   # Bulk volume of system..use this in ReactionSystem

Vₒ = 4.189e-09;  # volume of a singlet with 10 μm dia
integ(x) = Int(floor(x));
n = integ(N/2);
nr = N%2 == 0 ? (n*(n + 1) - n) : (n*(n + 1)); # No. of forward reactions

pair = [];
for i = 2:N
    push!(pair,[1:integ(i/2)  i .- (1:integ(i/2))])
end
pair = vcat(pair...);
vᵢ = @view pair[:,1];  # Reactant 1 index
vⱼ = @view pair[:,2];  # Reactant 2 index
volᵢ = Vₒ*vᵢ;    # cm⁻³
volⱼ = Vₒ*vⱼ;    # cm⁻³
sum_vᵢvⱼ = @. vᵢ + vⱼ;  # Product index

i = parse(Int, input("Enter 1 for additive kernel,
                2 for Multiplicative, 3 for constant"))
if i==1
    B = 1.53e03;    # s⁻¹
    kₛ = @. B*(volᵢ + volⱼ)/V;
elseif i==2
    b = 3.8e11;     #  cm⁻³ s⁻¹
    kₛ = @. b*(volᵢ*volⱼ)/V;
else
    C = 1.84e-04;   # cm³ s⁻¹
    kₛ = @. C/V;
end

## Writing-off the parameter in Pairs in Sequence
@variables k[1:nr];   pₘₐₚ = Pair.(k, kₛ);
@parameters t;        @variables X[collect(1:N)](t);
if i == 1
    tspan = (0. ,2000.)   # time-span
elseif i == 2
    tspan = (0. ,3000.)
else
    tspan = (0. ,350.)
end
u₀ = zeros(Int64, N);   u₀[1] = uₒ;   # initial condition of monomers
u₀map = Pair.(X, u₀); # population of other polymers in zeros
rx = [];              # empty-reaction vector

##  Forming ReactionSystem
@time for n = 1:nr
    if (vᵢ[n] == vⱼ[n])    # checking the reactants
        push!(rx, Reaction(2*k[n], [ X[vᵢ[n]] ] ,[ X[sum_vᵢvⱼ[n]] ] ,[2],[1]));
    else
        push!(rx, Reaction(k[n], [ X[vᵢ[n]] , X[vⱼ[n]] ] ,[ X[sum_vᵢvⱼ[n]] ],
                                [1, 1],[1]));
    end
end
rs = ReactionSystem(rx, t, X, k);

#=
odesys = convert(ODESystem, rs; combinatoric_ratelaws = true);
oprob = ODEProblem(odesys, u₀map, tspan, pₘₐₚ; parallel = true)
osol = solve(oprob, Tsit5())
=#

jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws = true);
dprob = DiscreteProblem(jumpsys, u₀map, tspan, pₘₐₚ; parallel = true);
stepper = SSAStepper()

## benchmarking the solvers.. Credits to SciMLBenchmarks (https://benchmarks.sciml.ai/)

methods = (Direct(), RDirect(), DirectCR(), DirectFW(), FRM(), FRMFW(),
            NRM(), RSSA(), RSSACR(), SortingDirect())

shortlabels = [string(leg) for leg in methods]

function run_benchmark!(t, jump_prob, stepper)
    sol = solve(jump_prob, stepper)
    @inbounds for i in 1:length(t)
        t[i] = @elapsed (sol = solve(jump_prob, stepper))
    end
end

nsims = 100
benchmarks = Vector{Vector{Float64}}()
for method in methods
    jprob = JumpProblem(jumpsys, dprob, method, save_positions=(false,false))
    t = Vector{Float64}(undef, nsims)
    run_benchmark!(t, jprob, stepper)
    push!(benchmarks, t)
end

medtimes = Vector{Float64}(undef,length(methods))
stdtimes = Vector{Float64}(undef,length(methods))
avgtimes = Vector{Float64}(undef,length(methods))
for i in 1:length(methods)
    medtimes[i] = median(benchmarks[i])
    avgtimes[i] = mean(benchmarks[i])
    stdtimes[i] = std(benchmarks[i])
end
using DataFrames

df = DataFrame(names=shortlabels, medtimes=medtimes, relmedtimes=(medtimes/medtimes[1]),
                avgtimes=avgtimes, std=stdtimes, cv=stdtimes./avgtimes)

sa = [string(round(mt,digits=4),"s") for mt in df.medtimes]
bar(df.names, df.relmedtimes, legend=:false)
scatter!(df.names, .05 .+ df.relmedtimes, markeralpha=0, series_annotations=sa)
ylabel!("median relative to Direct")
title!("Population Balance Equation")
