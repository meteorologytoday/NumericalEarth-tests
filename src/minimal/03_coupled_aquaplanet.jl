using Printf
@printf("Loading libraries...\n")
using NCDatasets

using SpeedyWeather, ConservativeRegridding
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ScalarDiffusivity, VerticallyImplicitTimeDiscretization

using NumericalEarth
@printf("Done\n")

@printf("Constructing atmosphere model... \n")
# Atmosphere model
spectral_grid = SpectralGrid(trunc=31, nlayers=4, Grid=FullGaussianGrid)
land_sea_mask = AquaPlanetMask(spectral_grid)
atmosphere = atmosphere_simulation(spectral_grid; output_interval=Hour(3))

@printf("Constructing ocean model...\n")
Nx, Ny, Nz = 60, 30, 8
grid = LatitudeLongitudeGrid(;
    size      = (Nx, Ny, Nz),
    longitude = (-180, 180),
    latitude  = (-75, 75),
    z         = (-2000meters, 0),
    halo      = (5, 5, 4),
)

ocean = ocean_simulation(
    grid;
    model = :hydrostatic,
    tracers             = (:T, :S),
    coriolis            = HydrostaticSphericalCoriolis(),
    momentum_advection  = WENOVectorInvariant(order=3),
    tracer_advection    = WENO(order=5),
    free_surface        = SplitExplicitFreeSurface(grid; substeps=30),
    closure             = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1e-4, κ=1e-5),
)

@inline T_target(y) = 28.0 * cosd(y)^2                    # target SST profile (°C)

Tᵢ(x, y, z) = T_target(y) * (1 + 0.01z / 2000)
Sᵢ(x, y, z) = 35.0
Oceananigans.set!(ocean.model, T=Tᵢ, S=Sᵢ)

#@printf("Contructing sea ice model... \n")
#sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=3))

@printf("Contructing coupled model... \n")
Δt = 2 * convert(eltype(grid), atmosphere.model.time_stepping.Δt_sec)
earth_model = EarthSystemModel(; atmosphere, ocean)

earth = Oceananigans.Simulation(earth_model; Δt, stop_time=15days)

@printf("Setup output writers for ocean... \n")
# atmosphere output is handled internally via output_interval=Hour(3) in atmosphere_simulation

save_fields_interval = 1days
u = ocean.model.velocities.u
v = ocean.model.velocities.v
w = ocean.model.velocities.w
filename_prefix = "output_momentum"
ocean.output_writers[:full] = Oceananigans.NetCDFWriter(
    ocean.model, (;u, v, w), filename=filename_prefix * ".nc",
    schedule = TimeInterval(save_fields_interval),
    overwrite_existing = true,
)

@printf("Setup output writers for coupling fluxes... \n")
sensible_heat = earth.model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
latent_heat = earth.model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
x_momentum = earth.model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
y_momentum = earth.model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
fluxes = (; sensible_heat, latent_heat, x_momentum, y_momentum)

ocean.output_writers[:fluxes] = Oceananigans.NetCDFWriter(
    earth.model.ocean.model, fluxes;
    overwrite_existing=true,
    schedule=TimeInterval(3hours),
    filename="intercomponent_fluxes.nc"
)

@printf("Running aquaplanet simulation...\n")
Oceananigans.run!(earth)
@printf("Done.\n")
