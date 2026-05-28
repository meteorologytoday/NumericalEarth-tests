# Fix: CairoMakie fails to load due to GLib version conflict

**Date:** 2026-05-26  
**System:** HPC (RHEL 9, glibc 2.34)  
**Julia version:** 1.12.6  

---

## Symptom

`using CairoMakie` (or any Julia startup that triggers `Glib_jll`) fails with:

```
ERROR: InitError: could not load library "...artifacts/.../lib/libgobject-2.0.so"
/home/tienyiao/.julia/artifacts/.../lib/libgobject-2.0.so: undefined symbol: g_string_copy
```

during initialization of module `Glib_jll`.

---

## Root Cause

Three factors combine to cause the failure:

| Factor | Detail |
|---|---|
| Julia artifact (`Glib_jll`) | v2.86.3 — bundles GLib 2.86.3 |
| System GLib (via `LD_LIBRARY_PATH`) | v2.81.0 at `/p/system/packages/libraries/glib/2.81.0/lib` |
| `g_string_copy` introduced | GLib 2.82 |

Julia's `libgobject-2.0.so` uses `RUNPATH=$ORIGIN` to find its companion `libglib-2.0.so.0`.
However, `RUNPATH` has **lower priority** than `LD_LIBRARY_PATH`.
The HPC module system sets `LD_LIBRARY_PATH` to include the system GLib 2.81.0 path, which is
found first — and that version is missing the `g_string_copy` symbol added in GLib 2.82.

---

## Fix

Downgrade `Glib_jll` in the Julia global environment to v2.80.x, which does not require
`g_string_copy` and is compatible with the system's GLib 2.81.0:

```julia
using Pkg
Pkg.add(name="Glib_jll", version="2.80")
```

This also downgrades the following transitive dependencies to compatible versions:

| Package | From | To |
|---|---|---|
| `Glib_jll` | v2.86.3 | v2.80.5 |
| `Cairo_jll` | v1.18.7 | v1.18.2 |
| `Pango_jll` | v1.57.1 | v1.54.1 |
| `HarfBuzz_jll` | v8.5.1 | v8.3.1 |
| `Pixman_jll` | v0.46.4 | v0.43.4 |
| `FFMPEG_jll` | v8.1.0 | v7.1.0 |
| `Libffi_jll` | v3.4.7 | v3.2.2 |

---

## Verification

After the downgrade, CairoMakie loads and saves figures correctly:

```julia
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, 1:10, rand(10))
save("test.png", fig)
```

A `GLib-CRITICAL: g_datalist_id_set_data_full` warning may still appear during
precompilation — this is cosmetic and does not affect functionality.

---

## Alternative Fixes (not applied)

- **Unload the GLib system module** before launching Julia:
  ```bash
  module unload glib
  julia
  ```
- **Prepend the artifact lib path** to `LD_LIBRARY_PATH` before launching Julia
  (fragile — artifact hash changes when packages are updated).
- **Use a different plotting backend** such as `GLMakie` (requires X11/display)
  or `PythonPlot` (requires Python + matplotlib).
