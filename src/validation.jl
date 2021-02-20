using ModelingToolkit, LinearAlgebra
using DiffEqBase ,DiffEqJump, OrdinaryDiffEq
using LoopVectorization, Plots
using BenchmarkTools
using SpecialFunctions, Statistics
plotly()

## Parameter
N = 5;         # Number of clusters
Vₒ = (4π/3)*(10e-06*100)^3;  # volume of a singlet in cm³
Nₒ = 1e-06/Vₒ;   # initial conc. = (No. of init. monomers) / Volume of the bulk system
uₒ = 10000;      # No. of singlets initially
V = uₒ/Nₒ;       # Bulk volume of system in cm³

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
volᵢ = Vₒ*vᵢ;
volⱼ = Vₒ*vⱼ;
sum_vᵢvⱼ = @. vᵢ + vⱼ; # Product index

i = parse(Int, input("Enter 1 for additive kernel,
        2 for Multiplicative, 3 for constant"))

if i==1
    B = 1.53e03;     # s⁻¹
    kₛ = @. B*(volᵢ + volⱼ)/V;
elseif i==2
    b = 3.80e11;     #  cm⁻³ s⁻¹
    kₛ = @. b*(volᵢ*volⱼ)/V;
else
    C = 1.80e-04;    #  cm³ s⁻¹
    kₛ = fill(C/V ,nr);
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

##  Forming ReactionSystem
rx = [];              # empty-reaction vector
reactant_stoich = Array{Array{Pair{Int64,Int64},1},1}(undef, nr);
net_stoich = Array{Array{Pair{Int64,Int64},1},1}(undef, nr);
@time for n = 1:nr
    if (vᵢ[n] == vⱼ[n])    # checking the reactants
        push!(rx, Reaction(2*k[n], [ X[vᵢ[n]] ] ,[ X[sum_vᵢvⱼ[n]] ] ,[2],[1]));

        reactant_stoich[n] = [vᵢ[n] => 2];
        net_stoich[n] = [vᵢ[n] => -2, sum_vᵢvⱼ[n] => 1];
    else
        push!(rx, Reaction(k[n], [ X[vᵢ[n]] , X[vⱼ[n]] ] ,[ X[sum_vᵢvⱼ[n]] ],
                                [1, 1],[1]));

        reactant_stoich[n] = [vᵢ[n] => 1 , vⱼ[n] => 1];
        net_stoich[n] = [vᵢ[n] => -1 , vⱼ[n] => -1 , sum_vᵢvⱼ[n] => 1];
    end
end
rs = ReactionSystem(rx, t, X, k);

##  Solving the reaction using different DiffEqJump solver
#=
odesys = convert(ODESystem, rs; combinatoric_ratelaws = true);
oprob = ODEProblem(odesys, u₀map, tspan, pₘₐₚ; parallel = true)
osol = solve(oprob, Tsit5())
=#
jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws = true);
dprob = DiscreteProblem(jumpsys, u₀map, tspan, pₘₐₚ; parallel = true);
alg = Direct();
stepper = SSAStepper();
mass_act_jump = MassActionJump(kₛ ,reactant_stoich, net_stoich);
jprob = @btime JumpProblem(dprob, alg ,mass_act_jump ,save_positions=(false,false));
jsol = @btime solve(jprob, stepper, saveat = 1.);

## Results for first three polymers
v_res = [1 ;2 ;3]         # comparsion with analytical solution
plot(jsol)
if i == 1
    ϕ = @. 1 - exp(-B*Nₒ*Vₒ*jsol.t);        # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
        sol[j,:] = @. Nₒ*(1 - ϕ)*(((j*ϕ)^(j-1))/gamma(j+1))*exp(-j*ϕ);
    end
elseif i == 2
    ϕ = @. (b*Nₒ*Vₒ*Vₒ*jsol.t);             # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
        sol[j,:] = @. Nₒ*(((j*ϕ)^(j-1))/(j*gamma(j+1)))*exp(-j*ϕ);
    end
else
    ϕ = @. (C*Nₒ*jsol.t);                   # normalised "time"
    sol = zeros(length(v_res), length(ϕ))
    for j in v_res
        sol[j,:] = @. 4Nₒ*((ϕ^(j-1))/((ϕ + 2)^(j+1)));
    end
end
# plotting normalised concentration vs analytical solution
plot(ϕ, jsol[1,:]/uₒ, lw = 2, xlabel = "Time (sec)",label = string("X",1),
    title = "Constant kernel")
plot!(ϕ, sol[1,:]/Nₒ, lw = 2, line = (:dot, 4) ,
        label = string("Analytical sol", "--X",1))

plot!(ϕ, jsol[2,:]/uₒ, lw = 2, xlabel = "Time (sec)",label = string("X",2))
plot!(ϕ, sol[2,:]/Nₒ, lw = 2, line = (:dot, 4) ,
        label = string("Analytical sol", "--X",2))

plot!(ϕ, jsol[3,:]/uₒ, lw = 2, xlabel = "Time (sec)",label = string("X",3))
plot!(ϕ, sol[3,:]/Nₒ, lw = 2, line = (:dot, 4) ,ylabel = "N(i,t)/N(1,0)"
        ,label = string("Analytical sol", "--X",3))
