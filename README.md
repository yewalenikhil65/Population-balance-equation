# Population balance equation
**Implementation of population balance equation** 
Smoluchowski coagulation equation is a population balance equation that describes the system of reactions in which ,say Singlets collide to form doublets, singlets and doublets collide to form triplets and so on. This models many chemical/physical processes such as polymerization, flocculation etc.

This is a short tutorial on implementation of [Smoluchowski coagulation equation](https://en.wikipedia.org/wiki/Smoluchowski_coagulation_equation) using [ModelingToolkit](https://mtk.sciml.ai/stable/)/[Catalyst](https://catalyst.sciml.ai/dev/) framework and it's comparison with analytical solution obtained by [Method of scotts](https://journals.ametsoc.org/view/journals/atsc/25/1/1520-0469_1968_025_0054_asocdc_2_0_co_2.xml)

  - **1.)**  Importing some important packages.
```julia
using ModelingToolkit, LinearAlgebra
using DiffEqBase ,DiffEqJump, OrdinaryDiffEq
using LoopVectorization, Plots
using BenchmarkTools
using SpecialFunctions
plotly()
```
  - **2.)**  Lets say there are `N` number of cluster size particles in the system. Lets initialise the system with some initial concentration `C`, initial number of singlets `uₒ` in the system. Since its a bimolecular chain of Reaction system(`nr` number of reactions), the bulk volume `V` of the system in which these binary collisions occur is important in the calculation of rate laws.
  
```julia
## Parameter
N = 5;      # Number of clusters
C  = 238.720;  # initial concentration == number of initial monomers/ volume of the bulk system
uₒ = 10000; # No. of singlets initially
V = uₒ/C;   # Bulk volume of system..use this in ReactionSystem

Vₒ = 4.189e-09;  # volume of a singlet with 10 μm dia
integ(x) = Int(floor(x));
n = integ(N/2);
nr = N%2 == 0 ? (n*(n + 1) - n) : (n*(n + 1)); # No. of forward reactions
```
  - **3.)**  Check the figure on [Smoluchowski coagulation equation](https://en.wikipedia.org/wiki/Smoluchowski_coagulation_equation) page, the `pair` of reactants that collide can be easily generated for `N` cluster size particles in the system. We also initialise the volumes of these colliding clusters as `volᵢ` and `volⱼ` for the reactants
  
```julia
## pairs of reactants
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
```
  - **4.)**  Specifying rate(kernel) at which reactants collide to form product. For simplicity we have used additive kernel, multiplicative kernel and constant kernel. The constants(`B`,`b` and `C`) used are adopted from the Scotts paper 
```julia
i = parse(Int, input("Enter 1 for additive kernel,
                2 for Multiplicative, 3 for constant"))
if i==1
    B = 1.53e03;    # s⁻¹
    kₛ = @. B*(volᵢ + volⱼ)/V;    # dividing by volume as its a bi-molecular reaction chain
elseif i==2
    b = 3.8e11;     #  cm⁻³ s⁻¹
    kₛ = @. b*(volᵢ*volⱼ)/V;
else
    C = 1.84e-04;   # cm³ s⁻¹
    kₛ = @. C/V;
end
```
  - **5.)**  Lets write-off the rates in `pₘₐₚ` as Pairs and initial condition with only singlets present initially in `u₀map` that we  will use in creating JumpSystems with massaction.
```julia
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
```
  - **6.)**  Push the Reactions(into empty reaction_network) as shown in figure at [here](https://en.wikipedia.org/wiki/Smoluchowski_coagulation_equation). When `vᵢ[n] == vⱼ[n]` ,we use rate as `2*k[n]` ,as coagulation kernel is related to deterministic rate-law in form, (to followed from a paper by [Laurenzi et.al](https://www.sciencedirect.com/science/article/pii/S0021999102970178))
          coagulation kernel = (1 + δᵢⱼ)*deterministic rate
                
```julia
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
```
  - **7.)**  Convert the reactionSystem into a JumpSystema and solve it using standard Jump solvers such as Gillespie process. For details, take a look at [DifferentialEquations](https://diffeq.sciml.ai/stable/) documentation 
```julia
## solving the system
jumpsys = convert(JumpSystem, rs; combinatoric_ratelaws = true);
dprob = DiscreteProblem(jumpsys, u₀map, tspan, pₘₐₚ; parallel = true);
alg = Direct();
jprob = @btime JumpProblem(jumpsys, dprob, alg);
jsol = @btime solve(jprob, SSAStepper());
```
  - **8.)**  Lets check the results for only first three polymers/cluster sizes. The result is compared with analytical solution obtained for this system with additive, multiplicative and constant kernels(rate at which reactants collide)
```julia
## Results for first three polymers
v_res = [1;2;3]

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
plot(ϕ, (jsol[1,:]), lw = 2, xlabel = "Time (sec)")                        # first species/singlet
plot!(ϕ, sol[1,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")

plot!(ϕ, (jsol[2,:]), lw = 2, xlabel = "Time (sec)")                      # doublets
plot!(ϕ, sol[2,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")

plot!(ϕ, (jsol[3,:]), lw = 2, xlabel = "Time (sec)")                      # triplets
plot!(ϕ, sol[3,:], lw = 2, line = (:dot, 4) ,label = "Analytical sol")

```
**Benchmarking PBE** - Credits to [SciMLBenchmarks](https://benchmarks.sciml.ai/) for the elegant codes for the purpose of benchmarking
click [here](https://github.com/yewalenikhil65/Population-balance-equation/blob/main/Figs/benchmarking_PBE.png) to see how different Jump algorithms fare for solving PBE stochastically
