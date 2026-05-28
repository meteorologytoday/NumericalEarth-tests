using SpeedyWeather, XESMF
using NumericalEarth
println(pkgversion(SpeedyWeather))
println(pkgversion(NumericalEarth))

spectral_grid = SpeedyWeather.SpectralGrid(; trunc=31, nlayers=4, Grid=FullGaussianGrid, dealiasing=3, architecture=SpeedyWeather.CPU())
atmosphere = atmosphere_simulation(spectral_grid; output=true)

show(err)
