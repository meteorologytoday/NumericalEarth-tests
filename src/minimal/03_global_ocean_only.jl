using Printf

@printf("Loading libraries...\n")
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Logging
using Base.Threads
@printf("Done\n")

@printf("Print out thread information:\n")
@printf("nthread = %d\n", nthreads())

@printf("Create grid and model... \n")
Nx = 60
Ny = 30
Nz = 4
depth = 100meters
z = (-depth, 0)
grid = TripolarGrid(CPU(); size = (Nx, Ny, Nz), halo = (7, 7, 4), z)

bottom_height(x, y) = - depth
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

ocean = ocean_simulation(grid, timestepper = :QuasiAdamsBashforth2)
model = OceanOnlyModel(ocean)

@printf("Creata simulation\n")
simulation = Simulation(model, Δt=20minutes, stop_time=1hour)
@printf("Ready to run simulation\n")
with_logger(NullLogger()) do

    @printf("Run simulation\n")
    run!(simulation)
end
model
