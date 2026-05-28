using Printf
using NCDatasets
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters
using CUDA

using Base.Threads

@printf("Print out thread information:\n")
@printf("nthread = %d\n", nthreads())

Δt=100seconds
stop_time = 360days

@printf("Grid generation...\n")
H = 1000meters
W = 1000kilometers
Nx, Ny, Nz = 100, 100, 50
#Nx, Ny, Nz = 32, 32, 10
W_x = W
W_y = W

grid = RectilinearGrid(
    GPU(),
    size=(Nx, Ny, Nz),
    x=(0, W_x),
    y=(0, W_y),
    z=(-H, 0),
    halo=(4,4,4),
    topology=(Periodic, Periodic, Bounded),
)

# A mountain in the ocean
mountain_h₀ = 250meters
mountain_width = 20kilometers
cylindar_radius = 490kilometers
x_center = W_x/2
y_center = W_y/2
#hill(x, y) = mountain_h₀ * exp(-( (x-x_center)^2 + (y-y_center)^2) / 2mountain_width^2)

#-H + hill(x, y)
bottom(x, y) = ((x-x_center)^2 + (y-y_center)^2 > cylindar_radius^2) ? 0.0 : - H

grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))
x = xnodes(grid, Center())
bottom_boundary = interior(grid.immersed_boundary.bottom_height, :, 1, 1)
top_boundary = 0 * x

@printf("Create boundary conditions\n")
const ρ0 = 1025.0
const cp = 3999.0
const τx = 0.1 # N/m^2 
const τy = 0.0
const Q = - 200.0
const F = 1e-7 # m/s ( 3 m/year )
const S0 = 35.0

Q_T(x, y, t) = Q / (ρ0 * cp)
S_flux(x, y, t) = -F * S0
τu(x, y, t) = τx / ρ0
τv(x, y, t) = τy / ρ0

boundary_conditions = (
    T = FieldBoundaryConditions(
        top = FluxBoundaryCondition(Q_T)
    ),
    S = FieldBoundaryConditions(
        top = FluxBoundaryCondition(S_flux)
    ),
    u = FieldBoundaryConditions(
        top = FluxBoundaryCondition(τu)
    ),
    v = FieldBoundaryConditions(
        top = FluxBoundaryCondition(τv)
    ),
)

@printf("Model creation\n")
model = HydrostaticFreeSurfaceModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    momentum_advection = WENO(),
    tracer_advection = WENO(),
    boundary_conditions = boundary_conditions,
)

@printf("Set initial conditions\n")
ϵ(x, y, z) = 2rand() - 1
Tᵢ(λ, φ, z) = 30 + z/H * 15
Sᵢ(λ, φ, z) = 28
set!(model, T=Tᵢ, S=Sᵢ, u=ϵ, v=ϵ)

@printf("Construct Simulation object\n")
simulation = Simulation(model; Δt=Δt, stop_time=stop_time)
progress(sim) = @printf("Iter: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
                        iteration(sim), time(sim), prettytime(sim.run_wall_time),
                        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

simulation.output_writers[:checkpointer] = Checkpointer(model, 
    schedule = TimeInterval(360days),
    prefix = "my_checkpoint",       # Filename prefix
    cleanup = false,                  # Keep only the most recent checkpoint
)

@printf("Setup output writer\n")
save_fields_interval = 5days
T = model.tracers.T
S = model.tracers.S
filename_prefix = "output_thermal"
simulation.output_writers[:full] = NetCDFWriter(
    model, (;T,S), filename=filename_prefix * ".nc",
    schedule = TimeInterval(save_fields_interval),
    overwrite_existing = true,
)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w
filename_prefix = "output_momentum"
simulation.output_writers[:full] = NetCDFWriter(
    model, (;u, v, w), filename=filename_prefix * ".nc",
    schedule = TimeInterval(save_fields_interval),
    overwrite_existing = true,
)


@printf("Model information:\n")
display(model)
@printf("Simulation information:\n")
display(simulation)
@printf("Run model\n")
run!(simulation)

exit()



#model = NonhydrostaticModel(; grid, advection=WENO())
#ϵ(x, y) = 2rand() - 1
#set!(model, u=ϵ, v=ϵ)
#simulation = Simulation(model; Δt=0.01, stop_time=4)
#run!(simulation)

using CairoMakie

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((0, grid.Lx/W), (-grid.Lz, 0)))

band!(ax, x/1e3, bottom_boundary, top_boundary, color = :mediumblue)

fig

