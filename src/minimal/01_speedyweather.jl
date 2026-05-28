using SpeedyWeather

# components
spectral_grid = SpectralGrid(trunc=31, nlayers=8)
ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)
land_sea_mask = AquaPlanetMask(spectral_grid)
orography = NoOrography(spectral_grid)

# create model, initialize, run
model = PrimitiveWetModel(spectral_grid; ocean, land_sea_mask, orography)
simulation = initialize!(model)
run!(simulation, period=Day(10))


using CairoMakie

humid = simulation.variables.grid.humidity[:, end]
fig = heatmap(humid, title="Surface specific humidity [kg/kg]", colormap=:oslo)
save("humidity.png", fig)

