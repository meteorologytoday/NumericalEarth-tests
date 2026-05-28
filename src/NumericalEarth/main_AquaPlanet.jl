using Printf

@printf("Loading Julia libraries...\n")

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

@printf("Libaries loaded.\n")

Δt=100seconds
stop_time = 360days

@printf("Ocean model setup.\n")

Nx = 30
Ny = 30
Nz = 10
z = ExponentialDiscretization(Nz, -2000, 0)
grid = TripolarGrid(Oceananigans.CPU(); size=(Nx, Ny, Nz), z, halo=(6, 6, 5))

momentum_advection = WENOVectorInvariant(order=3)
tracer_advection   = Centered()

free_surface = SplitExplicitFreeSurface(grid; substeps=30)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
                                       DiffusiveFormulation

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=DiffusiveFormulation())
vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()

closure = (eddy_closure, vertical_mixing)

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         closure,
                         free_surface)

# Initialize with uniform T≈15°C and S=35 PSU so SST ≈ 288 K matches the
# Jablonowski atmosphere surface temperature, avoiding a large initial air-sea ΔT.
# Also seed CATKE TKE to avoid 0/N² degeneracy with uniform stratification.
Oceananigans.set!(ocean.model, T=15.0, S=35.0, e=1e-6)

#@printf("Sea ice model setup.\n")
#sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=3))
sea_ice = NumericalEarth.default_sea_ice()

@printf("Atmosphere model setup.\n")
spectral_grid = SpeedyWeather.SpectralGrid(; trunc=31, nlayers=4, Grid=FullClenshawGrid, dealiasing=3, architecture=SpeedyWeather.CPU())
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
    uomax = (maximum(abs, interior(uo)), maximum(abs, interior(vo)), maximum(abs, interior(wo)))
    
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
