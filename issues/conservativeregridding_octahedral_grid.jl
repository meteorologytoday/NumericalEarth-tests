# Issue: ConservativeRegridding does not support OctahedralGaussianGrid
#
# When constructing a coupled atmosphere-ocean simulation using NumericalEarth
# with SpeedyWeather as the atmosphere, building the EarthSystemModel fails if
# SpectralGrid uses the default OctahedralGaussianGrid.
#
# Error:
#   Not implemented for OctahedralGaussianGrid{...}
#   @ ConservativeRegriddingRingGridsExt ~/.julia/packages/ConservativeRegridding/.../ext/ConservativeRegriddingRingGridsExt.jl:19
#
# Root cause:
#   ConservativeRegridding only implements treeify() for AbstractFullGrid types.
#   OctahedralGaussianGrid is a reduced grid (fewer points near the poles) and
#   is not a subtype of AbstractFullGrid, so the regridder used to couple
#   atmosphere and ocean grids cannot be constructed.
#
# Fix:
#   Pass Grid=FullGaussianGrid to SpectralGrid so the atmosphere uses a full
#   regular Gaussian grid, which ConservativeRegridding supports.

using SpeedyWeather
using ConservativeRegridding

# Wrong (default) — causes "Not implemented for OctahedralGaussianGrid" error:
# spectral_grid = SpectralGrid(trunc=31, nlayers=8)

# Correct — use FullGaussianGrid instead:
spectral_grid = SpectralGrid(trunc=31, nlayers=8, Grid=FullGaussianGrid)

# The FullGaussianGrid has equal numbers of points at every latitude ring,
# which is required by ConservativeRegridding to build the coupling regridder
# between the SpeedyWeather atmosphere grid and the Oceananigans ocean grid.
