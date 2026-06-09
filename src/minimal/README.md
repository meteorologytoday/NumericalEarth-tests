# Minimal Examples

Minimal working scripts for coupled atmosphere-ocean simulations using
SpeedyWeather and Oceananigans via NumericalEarth.

## Tested Versions

| Package              | Version  |
|----------------------|----------|
| Julia                | 1.12.6   |
| NumericalEarth       | 0.5.3    |
| Oceananigans         | 0.108.2  |
| SpeedyWeather        | 0.20.2   |
| ConservativeRegridding | 0.2.2  |
| NCDatasets           | 0.14.15  |

## Scripts

| Script | Description |
|--------|-------------|
| `01_speedyweather.jl` | Standalone SpeedyWeather aquaplanet (atmosphere only) |
| `02_aquaplanet.jl` | Standalone Oceananigans aquaplanet (ocean only, idealized forcing) |
| `03_coupled_aquaplanet.jl` | Coupled SpeedyWeather + Oceananigans via NumericalEarth |

## Running

Activate the project environment before running any script:

```
julia --project=<path-to-NumericalEarth-tests> <script>.jl
```

## Notes

### Flat orography in the coupled model (`03_coupled_aquaplanet.jl`)
`NumericalEarth.atmosphere_simulation()` does not expose an `orography` kwarg, so
`NoOrography` cannot be passed through it. The script works around this by constructing
the `PrimitiveWetModel` manually — replicating the internals of `atmosphere_simulation`
with the addition of `land_sea_mask` and `orography` kwargs:

```julia
orography     = NoOrography(spectral_grid)
land_sea_mask = AquaPlanetMask(spectral_grid)

atmosphere_model = SpeedyWeather.PrimitiveWetModel(
    spectral_grid;
    land_sea_mask, orography,
    surface_heat_flux, surface_humidity_flux,
    ocean = SpeedyWeather.PrescribedOcean(), sea_ice = nothing,
)
atmosphere = SpeedyWeather.initialize!(atmosphere_model)
SpeedyWeather.initialize!(atmosphere; output=true)
```

The parameterization tendencies are also pre-computed manually to mirror what
`atmosphere_simulation` does internally for coupling initialization.

## Known Issues

### `run!` ambiguity
When `SpeedyWeather` and `Oceananigans` are both loaded, `run!` becomes
ambiguous. Use `Oceananigans.run!(simulation)` explicitly.

### `NetCDFWriter` requires NCDatasets
`NetCDFWriter` is a package extension in Oceananigans and is only available
after `using NCDatasets`. Load it before constructing any `NetCDFWriter`.

### SpeedyWeather must use `FullGaussianGrid` for coupling
`ConservativeRegridding` (used by `NumericalEarthSpeedyWeatherExt` to regrid
between atmosphere and ocean grids) only supports `AbstractFullGrid` types.
SpeedyWeather's default `OctahedralGaussianGrid` is a reduced grid and is not
supported. Always pass `Grid=FullGaussianGrid` to `SpectralGrid`:

```julia
spectral_grid = SpectralGrid(trunc=31, nlayers=4, Grid=FullGaussianGrid)
```

### `NumericalEarthSpeedyWeatherExt` trigger packages
The extension that provides `atmosphere_simulation` is triggered by loading
both `SpeedyWeather` and `ConservativeRegridding` (not XESMF, despite older
versions of NumericalEarth listing it). Both must be explicit dependencies in
the project environment:

```julia
using SpeedyWeather, ConservativeRegridding
using NumericalEarth  # atmosphere_simulation now available
```
