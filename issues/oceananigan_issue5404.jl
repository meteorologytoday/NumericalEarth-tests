using Oceananigans
using CUDA

grid = RectilinearGrid(GPU();
    topology = (Bounded, Bounded, Bounded),
    size = (10, 10, 10),
    x = (0, 10),
    y = (-5, 5),
    z = (-100, 0)
)
@inline bottom(x, y) = -100
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

@inline relax(x, y, z, t, T, p) = - p.rate
params = (
    rate = 1/864000.0,
)

model = HydrostaticFreeSurfaceModel(
    grid;
    tracers = (:T,),
    forcing = (;
        T = Forcing(relax; parameters=params, field_dependencies=:T)
    ),
)
simulation = Simulation(model, Δt=0.01, stop_iteration=10)
run!(simulation)
