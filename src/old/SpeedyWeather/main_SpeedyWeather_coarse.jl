using Printf
using Dates
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputWriters

using SpeedyWeather
using XESMF           # required to trigger NumericalEarthSpeedyWeatherExt
using NumericalEarth

# ---- Atmosphere ----
# T21 FullGaussianGrid: ~5.6° horizontal resolution, 4 vertical layers.
spectral_grid = SpeedyWeather.SpectralGrid(;
    trunc        = 21,
    nlayers      = 4,
    Grid         = FullGaussianGrid,
    dealiasing   = 2,
    architecture = SpeedyWeather.CPU(),
)

# atmosphere_simulation registers NumericalEarth's surface-flux hooks
# (PrescribedOceanHeatFlux, PrescribedOceanHumidityFlux) on the model.
# output=true activates SpeedyWeather's own NetCDF output to run_XXXX/.
atmosphere = atmosphere_simulation(spectral_grid; output=true)

@printf("Atmosphere: T%d, %d layers, Δt = %.0f s\n",
        spectral_grid.trunc,
        spectral_grid.nlayers,
        atmosphere.model.time_stepping.Δt_sec)

# ---- Ocean ----
# Coarse global ocean: ~6° horizontal, 5 layers, CPU, 1000 m depth.
Nx, Ny, Nz = 60, 30, 5
z = ExponentialDiscretization(Nz, -1000, 0)
grid = TripolarGrid(Oceananigans.CPU(); size=(Nx, Ny, Nz), z, halo=(6, 6, 3))

# SplitExplicitFreeSurface extends the grid halo to ~substeps+2.
# With Ny=30 we need substeps < 28; use 20 which is safe and stable at coarse resolution.
free_surface = SplitExplicitFreeSurface(grid; substeps=20)
ocean = ocean_simulation(grid; free_surface)

# Idealized initial conditions: warm tropics, cold poles, cooling with depth.
Tᵢ(λ, φ, z) = 28 - abs(φ) / 90 * 22 + z / 1000 * 10
Sᵢ(λ, φ, z) = 35
Oceananigans.set!(ocean.model, T=Tᵢ, S=Sᵢ)

@printf("Ocean grid: %dx%dx%d (lon x lat x z)\n", Nx, Ny, Nz)

# ---- Coupled model ----
# AtmosphereOceanModel = EarthSystemModel(atmosphere, ocean, sea_ice=nothing).
# Emissivities set to zero to use idealised (not full) radiation.
radiation  = Radiation(ocean_emissivity=0.0, sea_ice_emissivity=0.0)
earth_model = AtmosphereOceanModel(atmosphere, ocean; radiation)

# Couple at 2× the atmosphere timestep (standard choice).
Δt   = 2 * convert(eltype(grid), atmosphere.model.time_stepping.Δt_sec)
earth = Oceananigans.Simulation(earth_model; Δt, stop_time=5days)

@printf("Coupled Δt = %.0f s\n", Δt)

# ---- Ocean output writer ----
ocean.output_writers[:surface] = JLD2Writer(
    ocean.model,
    merge(ocean.model.velocities, ocean.model.tracers);
    overwrite_existing = true,
    schedule = TimeInterval(6hours),
    filename = "ocean_surface_coarse.jld2",
    indices  = (:, :, grid.Nz),
)

# ---- Progress callback ----
wall_time = Ref(time_ns())
function progress(sim)
    atmos  = sim.model.atmosphere
    ocean  = sim.model.ocean
    ua, va = atmos.diagnostic_variables.dynamics.u_mean_grid,
             atmos.diagnostic_variables.dynamics.v_mean_grid
    uo, vo, wo = ocean.model.velocities

    step_time = 1e-9 * (time_ns() - wall_time[])
    @printf("time: %s | max|ua|: (%.1e, %.1e) m/s | max|uo|: (%.1e, %.1e, %.1e) m/s | wall: %s\n",
            prettytime(sim),
            maximum(abs, ua), maximum(abs, va),
            maximum(abs, uo), maximum(abs, vo), maximum(abs, wo),
            prettytime(step_time))
    wall_time[] = time_ns()
end

add_callback!(earth, progress, TimeInterval(1days))

@printf("Running coupled T21 atmosphere + coarse ocean for 5 days...\n")
Oceananigans.run!(earth)
@printf("Done.\n")
