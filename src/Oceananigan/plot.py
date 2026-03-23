import traceback
from datetime import datetime
import multiprocessing
from pathlib import Path

import pandas as pd

import xarray as xr

use_ipython = 'get_ipython' in globals()
print(f"use_ipython = {str(use_ipython)}")


import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean as cmo
import numpy as np
    
def plot_simulation(detail):

    output_dir = detail["output_dir"]
    file_paths = detail["file_paths"]
    index = detail["index"]
    phase = detail["phase"]
    result = dict(detail=detail, status="UNKNOWN")
    frame = index

    try:

        output_figures = [
            Path(output_dir) / f"frame-{index:05d}.png",
        ]

        if phase == "detect":
            result["needs_work"] = not all([output_figure.exists() for output_figure in output_figures])
            result["status"] = "detect_ok"
            return result

        # Load files
        plot_data = {
            k : xr.open_dataset(file_path).isel(time=frame)
            for k, file_path in file_paths.items()
        }

        levels_w = np.linspace(-1, 1, 51)
        levels_T = np.linspace(10, 30, 51)
        levels_S = np.linspace(30, 40, 51)

        try:

            print(f"Plotting frame = {frame:d}")
            fig, axes = plt.subplots(
                1, 3,
                figsize=(30, 10),
                subplot_kw={'box_aspect': 1},
                squeeze=False,
            )

            ax = axes.flatten()

            _data_w = plot_data["momentum"]["w"].isel(z_aaf=-3)
            w_sample_level = _data_w.coords["z_aaf"].to_numpy().item()
            
            _data_T = plot_data["thermal"]["T"].isel(z_aac=-1)
            _data_S = plot_data["thermal"]["S"].isel(z_aac=-1)
            tracer_sample_level = _data_T.coords["z_aac"].to_numpy().item()
            
            coords = plot_data["momentum"].coords
            time_str = f"{str(_data_w['time'].to_numpy().item()/1e9/86400)} days"
            x_bnds = coords["x_faa"].to_numpy() /1e3
            y_bnds = coords["y_afa"].to_numpy() /1e3
            
            def pcolormesh(ax, data, levels, cmap, extend):
                norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
                mappable = ax.pcolormesh(
                    x_bnds, y_bnds,
                    data,
                    norm=norm,
                    cmap=cmap,
                )
                cb = plt.colorbar(ax=_ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.1)
                return cb

            ax_i = 0
            _ax = ax[ax_i] ; ax_i+=1
            cb = pcolormesh(_ax, _data_w * 1e4, levels=levels_w, cmap=cmo.cm.balance, extend="both")
            cb.set_label("[$10^{-4} \\, \\mathrm{m} \\, \\mathrm{s}^{-1}$]", fontsize=12)
            _ax.set_title(f"(a) Vertical velocity at z={w_sample_level:.1f}m")

            _ax = ax[ax_i] ; ax_i+=1
            cb = pcolormesh(_ax, _data_T, levels=levels_T, cmap=cmo.cm.thermal, extend="both")
            cb.set_label("[${}^{\\circ}\\mathrm{C}$]", fontsize=12)
            _ax.set_title(f"(b) Temperature at z={tracer_sample_level:.1f}m")

            _ax = ax[ax_i] ; ax_i+=1
            cb = pcolormesh(_ax, _data_S, levels=levels_S, cmap=cmo.cm.haline, extend="both")
            cb.set_label("[$\\mathrm{PSU}$]", fontsize=12)
            _ax.set_title(f"(c) Salinity at z={tracer_sample_level:.1f}m")



            for _ax in ax:
                _ax.set_xlabel("X [km]")
                _ax.set_ylabel("Y [km]")

            fig.suptitle(f"[{time_str:s}]")
            
            print("Saving figure: ", output_figures[0])
            fig.savefig(output_figures[0], dpi=200)
            plt.close(fig)

        except Exception as e:

            traceback.print_exc()
            result["status"] = "ERROR_individual_frame"

    except Exception as e:

        traceback.print_exc()
        result["status"] = "ERROR"

    return result

if __name__ == "__main__": 

    num_cores = multiprocessing.cpu_count()
    print(f"Parallelizing across {num_cores} cores...")
   

    input_args = [] 

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_indices = np.arange(563)

    phase = 'detect'
    for index in plot_indices:
        detail = dict(
            output_dir = str(output_dir),
            file_paths = dict(
                momentum = Path("output_momentum.nc"),
                thermal = Path("output_thermal.nc"),
            ),
            index = index,
            phase = phase,
        )
        
        result = plot_simulation(detail)
        if result["needs_work"]:
            detail["phase"] = "work"
            input_args.append(detail)
        else:
            print(f"Index={index} does not need work.")

    with multiprocessing.Pool(processes=num_cores) as pool:
        # pool.map distributes the frame indices to the plotting function
        pool.map(plot_simulation, input_args)
        
    print("All frames generated in 'figures/' folder.")



