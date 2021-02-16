# doing validation with analytical expression
using ModelingToolkit, LinearAlgebra
using DiffEqBase ,DiffEqJump, OrdinaryDiffEq
using LoopVectorization, Plots
using BenchmarkTools
using SpecialFunctions
plotly()
## Parameter
N = 5;      # Number of clusters
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
alg = RSSA();
jprob = @btime JumpProblem(jumpsys, dprob, alg);
jsol = @btime solve(jprob, SSAStepper());

## Results for first three polymers
v_res = [1;2;3]
plot(jsol , lw = 2, vars = v_res ,xlabel = "Time (sec)" )

## comparsion with analytical solution
if i == 1
    ϕ = @. 1 - exp(-0.00153*jsol.t) # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
         sol[j,:] = @. uₒ*(1 - ϕ)*(((j*ϕ)^(j-1))/gamma(j+1))*exp(-j*ϕ);
    end
elseif i == 2
    ϕ = @. (0.00159*jsol.t);   # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
        sol[j,:] = @. uₒ*(((j*ϕ)^(j-1))/(j*gamma(j+1)))*exp(-j*ϕ);
    end
else
    ϕ = @. (0.0429*jsol.t);  # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
        sol[j,:] = @. 4uₒ*((ϕ^(j-1))/((ϕ + 2)^(j+1)));
    end
end
# plotting normalised concentration vs analytical solution
plot(ϕ, (jsol[1,:]), lw = 2, xlabel = "Time (sec)")
plot!(ϕ, sol[1,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")

plot(ϕ, (jsol[2,:]), lw = 2, xlabel = "Time (sec)")
plot!(ϕ, sol[2,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")

plot(ϕ, (jsol[3,:]), lw = 2, xlabel = "Time (sec)")
plot!(ϕ, sol[3,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")