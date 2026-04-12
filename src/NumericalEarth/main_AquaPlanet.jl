using Printf
using Statistics
using Dates

using CUDA
using NCDatasets

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters

using XESMF
using SpeedyWeather
using NumericalEarth

Δt=100seconds
stop_time = 360days

@printf("Ocean model setup.\n")
H = 1000meters
Nx, Ny, Nz = 60, 30, 10
z = ExponentialDiscretization(Nz, -H, 0)
bottom(x, y) = - H

grid = TripolarGrid(
    Oceananigans.CPU();
    size=(Nx, Ny, Nz),
    z,
    halo=(3, 3, 3),
)
#grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

momentum_advection   = VectorInvariant()
tracer_advection     = WENO(order=3)
free_surface         = SplitExplicitFreeSurface(grid; substeps=40)
catke_closure        = NumericalEarth.Oceans.default_ocean_closure()
eddy_closure         = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)
viscous_closure      = Oceananigans.TurbulenceClosures.HorizontalScalarBiharmonicDiffusivity(ν=1e12)
closures             = (catke_closure, eddy_closure, viscous_closure)
ocean = ocean_simulation(
    grid;
    momentum_advection,
    tracer_advection,
    free_surface,
    closure = closures,
)

@printf("Sea ice model setup.\n")
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=3))

@printf("Atmosphere model setup.\n")
#spectral_grid = SpeedyWeather.SpectralGrid(; trunc=63, nlayers=4, Grid=FullClenshawGrid)
spectral_grid = SpeedyWeather.SpectralGrid(; trunc=31, nlayers=4, Grid=FullGaussianGrid, dealiasing=3, architecture=SpeedyWeather.CPU())
atmosphere = atmosphere_simulation(spectral_grid; output=true)
atmosphere.model.output.output_dt = Hour(3)

@printf("Radiation setup.\n")
radiation = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)

@printf("Couple models.\n")
earth_model = EarthSystemModel(atmosphere, ocean, sea_ice; radiation)
earth_model_Δt = 2 * convert(eltype(grid), atmosphere.model.time_stepping.Δt_sec)
earth = Oceananigans.Simulation(earth_model; Δt=earth_model_Δt, stop_time=5days)

@printf("earth_model_Δt = %f\n", earth_model_Δt)


@printf("Setup output writers.")
outputs = merge(ocean.model.velocities, ocean.model.tracers)
#=
sea_ice_fields = merge(
    sea_ice.model.velocities,
    sea_ice.model.dynamics.auxiliaries.fields,
    (; h=sea_ice.model.ice_thickness, ℵ=sea_ice.model.ice_concentration)
)
=#
ocean.output_writers[:free_surf] = JLD2Writer(ocean.model, (; η=ocean.model.free_surface.displacement);
                                              overwrite_existing=true,
                                              schedule=TimeInterval(3hours),
                                              including = [:grid],
                                              filename="ocean_free_surface.nc")

ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                            overwrite_existing=true,
                                            schedule=TimeInterval(3hours),
                                            including = [:grid],
                                            filename="ocean_surface_fields.nc",
                                            indices=(:, :, grid.Nz))
#=
sea_ice.output_writers[:fields] = JLD2Writer(sea_ice.model, sea_ice_fields;
                                             overwrite_existing=true,
                                             schedule=TimeInterval(3hours),
                                             including = [:grid],
                                             filename="sea_ice_fields.nc")
=#
𝒬ᵀᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
𝒬ᵛᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.latent_heat
τˣᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.x_momentum
τʸᵃᵒ = earth.model.interfaces.atmosphere_ocean_interface.fluxes.y_momentum
fluxes = (; 𝒬ᵀᵃᵒ, 𝒬ᵛᵃᵒ, τˣᵃᵒ, τʸᵃᵒ)
#=
𝒬ᵀᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat
𝒬ᵛᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.latent_heat
τˣᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.x_momentum
τʸᵃⁱ = earth.model.interfaces.atmosphere_sea_ice_interface.fluxes.y_momentum
𝒬ⁱᵒ  = earth.model.interfaces.net_fluxes.sea_ice.bottom.heat
Jˢⁱᵒ  = earth.model.interfaces.sea_ice_ocean_interface.fluxes.salt
fluxes = (; 𝒬ᵀᵃᵒ, 𝒬ᵛᵃᵒ, τˣᵃᵒ, τʸᵃᵒ, 𝒬ᵀᵃⁱ, 𝒬ᵛᵃⁱ, τˣᵃⁱ, τʸᵃⁱ, 𝒬ⁱᵒ, Jˢⁱᵒ)
=#

ocean.output_writers[:fluxes] = JLD2Writer(earth.model.ocean.model, fluxes;
                                           overwrite_existing=true,
                                           schedule=TimeInterval(3hours),
                                           including = [:grid],
                                           filename="intercomponent_fluxes.nc")

# We also add a callback function that prints out a helpful progress message while the simulation runs.
wall_time = Ref(time_ns())
function progress(sim)
    atmos = sim.model.atmosphere
    ocean = sim.model.ocean
    
    ua, va     = atmos.diagnostic_variables.dynamics.u_mean_grid, atmos.diagnostic_variables.dynamics.v_mean_grid
    uo, vo, wo = ocean.model.velocities
    
    uamax = (maximum(abs, ua), maximum(abs, va))
    uomax = (maximum(abs, uo), maximum(abs, vo), maximum(abs, wo))
    
    step_time = 1e-9 * (time_ns() - wall_time[])
    
    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|ua|: (%.1e, %.1e) m s⁻¹", uamax...)
    msg3 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", uomax...)
    msg4 = @sprintf(", wall time: %s \n", prettytime(step_time))
    
    @info msg1 * msg2 * msg3 * msg4
    wall_time[] = time_ns()
    
    return nothing
end

add_callback!(earth, progress, TimeInterval(1days))

@printf("Run the coupled model")
Oceananigans.run!(earth)
