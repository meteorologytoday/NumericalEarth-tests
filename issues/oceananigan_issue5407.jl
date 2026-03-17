using Printf
using Oceananigans
using Oceananigans.Units

grid = RectilinearGrid(
    CPU(),
    size=(5, 5, 2),
    x=(0, 100kilometers),
    y=(0, 100kilometers),
    z=(-100meters.0, 0),
    halo=(4,4,4),
    topology=(Bounded, Bounded, Bounded),
)

@inline function heat_source(x, y, z, t, T, p)
    @printf("T(x=%f, y=%f, z=%f, t=%f) = %f\n", x, y, z, t, T)
    return - T / p.τ_T
end

@inline function salt_source(x, y, z, t, S, p)
    @printf("S(x=%f, y=%f, z=%f, t=%f) = %f\n", x, y, z, t, S)
    return - S / p.τ_S
end

T_forcing = Forcing(heat_source, parameters=(;τ_T=86400.0*30), field_dependencies=:T)
S_forcing = Forcing(salt_source, parameters=(;τ_S=86400.0*360), field_dependencies=:S)

model = HydrostaticFreeSurfaceModel(
    grid;
    buoyancy = SeawaterBuoyancy(),
    tracers = (:T, :S),
    momentum_advection = WENO(),
    tracer_advection = WENO(),
    forcing=(T=T_forcing, S=S_forcing),
)

set!(model, T=30.0, S=28.5)
simulation = Simulation(model; Δt=100seconds, stop_iteration=10)
run!(simulation)

