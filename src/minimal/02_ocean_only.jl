using Printf

@printf("Loading libraries...\n")
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Logging
@printf("Done\n")



@printf("Create grid and model... \n")
grid = RectilinearGrid(size=10, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, timestepper = :QuasiAdamsBashforth2)
model = OceanOnlyModel(ocean)

@printf("Creata simulation\n")
simulation = Simulation(model, Δt=20minutes, stop_time=1hour)
@printf("Run simulation\n")
with_logger(NullLogger()) do
    run!(simulation)
end
model
