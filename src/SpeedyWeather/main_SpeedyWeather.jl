using Printf
using Statistics
using Dates

using CUDA
using NCDatasets

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters

using SpeedyWeather

@printf("Atmosphere model setup.\n")
spectral_grid = SpeedyWeather.SpectralGrid(; trunc=31, nlayers=4, Grid=FullGaussianGrid, dealiasing=3)
#atmosphere = atmosphere_simulation(spectral_grid, output=true)
#atmosphere.model.output.output_dt = Hour(3)


#=
spectral_grid = SpectralGrid(trunc=31, nlayers=8, Grid=FullGaussianGrid, dealiasing=3)

orography = ZonalRidge(spectral_grid)
initial_conditions = InitialConditions(
    vordiv = ZonalWind(spectral_grid),
    temp = JablonowskiTemperature(spectral_grid),
    pres = ConstantPressure(spectral_grid))

model = PrimitiveDryModel(spectral_grid; orography, initial_conditions, physics=false)
simulation = initialize!(model)
run!(simulation, period=Day(9))
=#
