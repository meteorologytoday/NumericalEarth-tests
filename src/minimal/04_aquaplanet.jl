using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ScalarDiffusivity, VerticallyImplicitTimeDiscretization

@printf("Loading libraries...\n")

# ── Grid ───────────────────────────────────────────────────────────────────────
# Low-resolution global aquaplanet: ~6° horizontal, 8 vertical levels, flat bottom
Nx, Ny, Nz = 60, 30, 8

grid = LatitudeLongitudeGrid(CPU();
    size      = (Nx, Ny, Nz),
    longitude = (-180, 180),
    latitude  = (-75, 75),
    z         = (-2000meters, 0),
    halo      = (5, 5, 4),
)

# ── Idealized surface forcing ──────────────────────────────────────────────────
# Wind stress: τ ~ -sin(2φ) gives trade winds (equatorial easterlies) + mid-lat westerlies
# Heat flux:   relax SST toward T*(φ) = 28°C * cos²(φ) on a 30-day timescale

const ρ₀       = 1025.0   # kg/m³, reference density
const τ₀       = 0.1      # N/m², peak wind stress amplitude
const τ_T      = 30days   # SST restoring timescale
const h_piston = 10.0     # m, effective piston depth (sets restoring flux magnitude)

@inline τx(x, y, t)          = -τ₀ * sin(2 * deg2rad(y)) / ρ₀     # kinematic wind stress (m²/s²)
@inline T_target(y)           = 28.0 * cosd(y)^2                    # target SST profile (°C)
@inline T_flux(x, y, t, T)   = -(T - T_target(y)) * h_piston / τ_T # restoring flux (K m/s)

# ── Boundary conditions ────────────────────────────────────────────────────────
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(T_flux; field_dependencies=:T))

# ── Model ──────────────────────────────────────────────────────────────────────
model = HydrostaticFreeSurfaceModel(
    grid;
    buoyancy            = SeawaterBuoyancy(),
    tracers             = (:T, :S),
    coriolis            = HydrostaticSphericalCoriolis(),
    momentum_advection  = WENOVectorInvariant(order=3),
    tracer_advection    = WENO(order=5),
    free_surface        = SplitExplicitFreeSurface(grid; substeps=30),
    closure             = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1e-4, κ=1e-5),
    boundary_conditions = (u=u_bcs, T=T_bcs),
)

# ── Initial conditions ─────────────────────────────────────────────────────────
# Temperature: target SST profile with slight depth stratification
# Salinity: uniform 35 PSU (no freshwater forcing in this setup)
Tᵢ(x, y, z) = T_target(y) * (1 + 0.01z / 2000)
Sᵢ(x, y, z) = 35.0
set!(model, T=Tᵢ, S=Sᵢ)

# ── Simulation ─────────────────────────────────────────────────────────────────
simulation = Simulation(model; Δt=20minutes, stop_time=30days)

wizard = TimeStepWizard(cfl=0.2, max_Δt=20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ── Output writers ─────────────────────────────────────────────────────────────
surface_fields = merge(model.velocities, model.tracers)

simulation.output_writers[:surface] = JLD2Writer(
    model, surface_fields;
    overwrite_existing = true,
    schedule           = TimeInterval(6hours),
    filename           = "aquaplanet_surface.jld2",
    indices            = (:, :, grid.Nz),
)

#=
simulation.output_writers[:free_surf] = JLD2Writer(
    model, (; η=model.free_surface.displacement);
    overwrite_existing = true,
    schedule           = TimeInterval(6hours),
    filename           = "aquaplanet_free_surface.jld2",
)
=#

@printf("Setup output writer\n")
save_fields_interval = 1days
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


# ── Progress callback ──────────────────────────────────────────────────────────
wall_time = Ref(time_ns())
function progress(sim)
    u, v = model.velocities.u, model.velocities.v
    T    = model.tracers.T
    step_time = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("time: %s, iter: %d, Δt: %s | max|u|: (%.1e, %.1e) m/s | SST: [%.1f, %.1f] °C | wall: %s",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   maximum(abs, interior(u)), maximum(abs, interior(v)),
                   minimum(interior(T, :, :, grid.Nz)), maximum(interior(T, :, :, grid.Nz)),
                   prettytime(step_time))
    wall_time[] = time_ns()
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

# ── Run ────────────────────────────────────────────────────────────────────────
@printf("Running aquaplanet simulation...\n")
run!(simulation)
@printf("Done.\n")
