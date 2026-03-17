using Printf
using NCDatasets
using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters
using CUDA

using Base.Threads

@printf("Print out thread information:\n")
@printf("nthread = %d\n", nthreads())

Δt=300seconds
stop_time = 100days

@printf("Grid generation...\n")
H = 1000meters
W = 1000kilometers
#Nx, Ny, Nz = 100, 100, 50

Nx, Ny, Nz = 10, 10, 10
W_x = W
W_y = W

@printf("Grid size: (%d, %d, %d)\n", Nx, Ny, Nz)

grid = RectilinearGrid(
    CPU(),
    size=(Nx, Ny, Nz),
    x=(0, W_x),
    y=(0, W_y),
    z=(-H, 0),
    halo=(4,4,4),
    topology=(Bounded, Bounded, Bounded),
)

# Make domain into a cylindar
cylindar_radius = 450kilometers
x_center = W_x/2
y_center = W_y/2
#bottom(x, y) = ((x-x_center)^2 + (y-y_center)^2 > cylindar_radius^2) ? 0.0 : - H
#grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

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

@printf("Add source and sink\n")
parameters = (;
    τ=10days,
    x_center=W_x/2,
    y_center=W_y/2,
    boundary_forcing_radius=450kilometers,
    center_forcing_radius=100kilometers,
    H=H,
)

#@inline function heat_source(x, y, z, t, T, p)
#    distance_square = (x-p.x_center)^2 + (y-p.y_center)^2
#    mask = distance_square > p.boundary_forcing_radius^2
#    return mask * (-(T - 25.0) * exp(z / p.H) / p.τ)
#end

@inline function heat_source(x, y, z, t, T, p)
    @printf("T = %f\n", T)
    return - (T - 20) / p.τ
end


@inline function salt_source(x, y, z, t, S, p)
    distance_square = (x-p.x_center)^2 + (y-p.y_center)^2
    
    if (distance_square > p.boundary_forcing_radius^2)
        return - (S - 40.0) * exp(z/p.H) / p.τ
    elseif (distance_square < p.center_forcing_radius^2)
        return - (S - 35.0) * exp(z/p.H) / p.τ
    else
        return 0.0
    end
end

S_forcing = Forcing(salt_source, parameters=parameters, field_dependencies=:S)
T_forcing = Forcing(heat_source, parameters=parameters, field_dependencies=:T)

@printf("Model creation\n")
model = HydrostaticFreeSurfaceModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    momentum_advection = WENO(),
    tracer_advection = WENO(),
#    boundary_conditions = boundary_conditions,
    forcing=(T=T_forcing,),# S=S_forcing,),
)

@printf("Set initial conditions\n")
ϵ(x, y, z) = 0.0 #2rand() - 1
Tᵢ(λ, φ, z) = 30 #+ z/H * 15
Sᵢ(λ, φ, z) = 28
set!(model, T=Tᵢ, S=Sᵢ, u=ϵ, v=ϵ)

@printf("Construct Simulation object\n")
simulation = Simulation(model; Δt=Δt, stop_time=stop_time)
progress(sim) = @printf("Iter: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
                        iteration(sim), time(sim), prettytime(sim.run_wall_time),
                        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

simulation.output_writers[:checkpointer] = Checkpointer(model, 
    schedule = TimeInterval(360days),
    prefix = "my_checkpoint",       # Filename prefix
    cleanup = false,                # Keep only the most recent checkpoint
)

@printf("Setup output writer\n")
save_fields_interval = 5days
T = model.tracers.T
S = model.tracers.S
filename_prefix = "output_thermal"
simulation.output_writers[:thermal] = NetCDFWriter(
    model, (;T,S), filename=filename_prefix * ".nc",
    schedule = TimeInterval(save_fields_interval),
    overwrite_existing = true,
)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w
filename_prefix = "output_momentum"
simulation.output_writers[:momentum] = NetCDFWriter(
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

