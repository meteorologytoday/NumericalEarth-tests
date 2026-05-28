using SpeedyWeather
spectral_grid = SpectralGrid(trunc = 31, nlayers = 8, Grid=FullGaussianGrid, dealiasing = 3)

orography = ZonalRidge(spectral_grid)
initial_conditions = (;                             # collect initial conditions into NamedTuple (keys don't matter)
    vordiv = ZonalWind(spectral_grid),
    temperature = JablonowskiTemperature(spectral_grid),
    pressure = ConstantPressure(spectral_grid))

model = PrimitiveDryModel(spectral_grid; orography, initial_conditions, dynamics_only = true)
simulation = initialize!(model)
run!(simulation, period = Day(9))
