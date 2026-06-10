# ---------------- Subprocess CAMB .npz wrapper with robust logging ----------------
import os
import sys
import tempfile
import shutil
import pickle
import subprocess
import time

import numpy as np


def compute_camb_glass_in_child_npz_subproc(
    param_dict,
    lmax,
    zmin,
    zmax,
    dx,
    mem_limit_gb: float = 50.0,
    timeout_s: int = 1800,
    sim_tag: str = None,
    tmpdir_base: str = "/share/gpu5/asaoulis/tmp",
):
    """Spawn a worker subprocess that recomputes CAMB and runs the lighter
    get_camb_matter_cls pipeline, saving results into an .npz-style pickle file.

    The worker will:
      * rebuild cosmology & CAMB from param_dict
      * call cosmology.camb_matter_power.get_camb_matter_cls
      * pickle {"shells": shells, "glass_cls": glass_cls} to out_npz

    Returns: absolute path to the output file on success.
    Raises: TimeoutError or RuntimeError (including child's traceback if available).
    """
    # create tempdir
    prefix = f"camb_npz_subproc_{sim_tag or int(time.time())}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=tmpdir_base)
    out_npz = os.path.join(tmpdir, "camb_child_outputs.npz")
    inputs_pkl = os.path.join(tmpdir, "inputs.pkl")
    worker_py = os.path.join(tmpdir, "camb_child_worker.py")
    stdout_log = os.path.join(tmpdir, "worker.stdout.log")
    stderr_log = os.path.join(tmpdir, "worker.stderr.log")
    errfile = out_npz + ".err"

    try:
        # 1) Serialize all inputs into inputs.pkl so the worker doesn't need argument pickling
        with open(inputs_pkl, "wb") as f:
            pickle.dump(
                {
                    "param_dict": param_dict,
                    "lmax": int(lmax),
                    "zmin": float(zmin),
                    "zmax": float(zmax),
                    "dx": float(dx),
                    "mem_limit_gb": float(mem_limit_gb),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # 2) Write worker script into tmpdir (structure mirrors levin_child_worker in mpi.py)
        worker_code = r'''
#!/usr/bin/env python3
import pickle
import sys
import os
import traceback
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: worker.py <inputs_pkl> <out_npz>", file=sys.stderr)
        sys.exit(2)

    inputs_pkl = sys.argv[1]
    out_npz = sys.argv[2]

    try:
        with open(inputs_pkl, "rb") as f:
            data = pickle.load(f)

        param_dict = data["param_dict"]
        lmax = int(data["lmax"])
        zmin = float(data["zmin"])
        zmax = float(data["zmax"])
        dx = float(data["dx"])

        # --- imports: identical pattern to levin worker ---
        import parameters as _parameters
        import camb
        from camb_matter_power import get_camb_matter_cls as _get_camb_matter_cls

        # --- rebuild CAMB state explicitly ---
        cosmo, pars = _parameters.build_cosmology(param_dict)

        # force CAMB initialization (matches levin behavior)
        results = camb.get_results(pars)
        results.calc_power_spectra(pars)

        # --- run CAMB matter power pipeline ---
        shells, glass_cls = _get_camb_matter_cls(
            pars, lmax, zmin, zmax, dx
        )

        # --- save outputs ---
        with open(out_npz, "wb") as f:
            pickle.dump(
                {
                    "shells": shells,
                    "glass_cls": glass_cls,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # fsync directory (same as levin worker)
        try:
            d = os.path.dirname(out_npz) or "."
            fd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass

        sys.exit(0)

    except MemoryError:
        tb = traceback.format_exc()
        with open(out_npz + ".err", "w") as ef:
            ef.write("MemoryError in worker:\n")
            ef.write(tb)
        print("MemoryError in worker; wrote .err", file=sys.stderr)
        sys.exit(3)

    except Exception:
        tb = traceback.format_exc()
        with open(out_npz + ".err", "w") as ef:
            ef.write(tb)
        print("Worker exception; wrote .err", file=sys.stderr)
        print(tb, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        with open(worker_py, "w") as wf:
            wf.write(worker_code)
        os.chmod(worker_py, 0o755)

        # 3) Launch the worker subprocess (use same interpreter), mimic mpi.py env setup
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_python = os.path.join(conda_prefix, "bin", "python")
        else:
            conda_python = sys.executable  # fallback

        cmd = [conda_python, worker_py, inputs_pkl, out_npz]
        env = os.environ.copy()
        # Match mpi.py: point PYTHONPATH just to cosmology package directory
        extra_path = "/home/asaoulis/projects/glass_transfer/src/cosmology"
        env["PYTHONPATH"] = extra_path + ":" + env.get("PYTHONPATH", "")

        # Always create explicit log files so we can inspect what happened
        with open(stdout_log, "wb") as outfh, open(stderr_log, "wb") as errfh:
            proc = subprocess.Popen(cmd, stdout=outfh, stderr=errfh, cwd=tmpdir, env=env)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"CAMB worker timed out after {timeout_s} s (tag={sim_tag}). See logs in {tmpdir}")

        # Inspect exit code and produce helpful errors
        exitcode = proc.returncode
        if exitcode != 0:
            if os.path.exists(errfile):
                with open(errfile, "r") as ef:
                    child_tb = ef.read()
                raise RuntimeError(
                    f"CAMB worker failed (exitcode={exitcode}). Traceback from worker (.err):\n{child_tb}\n\nSee logs: {stdout_log}, {stderr_log}"
                )
            else:
                try:
                    with open(stderr_log, "r") as ef:
                        stderr_txt = ef.read()
                except Exception:
                    stderr_txt = "<could not read stderr log>"
                raise RuntimeError(
                    f"CAMB worker exited with code {exitcode}. Stderr:\n{stderr_txt}\nSee logs in {tmpdir}"
                )

        if not os.path.exists(out_npz):
            raise RuntimeError(
                f"CAMB worker returned exitcode 0 but output file missing at {out_npz}. See logs in {tmpdir}"
            )

        return os.path.abspath(out_npz)

    except Exception:
        # keep tmpdir for debugging
        raise


# ---------------- loader for CAMB worker output ----------------
import os as _os
import pickle as _pickle
import shutil as _shutil


def load_camb_child_pickle(out_path, remove_after_load: bool = True):
    """Load pickle file produced by compute_camb_glass_in_child_npz_subproc worker.

    Returns (shells, glass_cls).
    """
    try:
        with open(out_path, "rb") as f:
            data = _pickle.load(f)
        shells = data["shells"]
        glass_cls = data["glass_cls"]
    except Exception as e:
        raise RuntimeError(f"Failed to load CAMB child pickle {out_path}: {e}")

    if remove_after_load:
        try:
            tmpdir = _os.path.dirname(out_path)
            _shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return shells, glass_cls


# ---------------- Persistent per-sim_id CAMB-Cls disk cache + guard ----------------
from pathlib import Path as _Path

# Cosmology parameters that fully determine the CAMB matter Cls (everything else in a
# param_dict — a_ia/b_ia and CAMB defaults — does not change the spectra). These are the keys
# the guard compares between a deterministically re-derived cosmology and a cached one.
COSMO_GUARD_KEYS = ("omega_m", "sigma_8", "ombh2", "h", "ns", "w0", "mnu")
# Grid parameters the CAMB Cls also depend on; cached spectra are only valid for a matching grid.
GRID_GUARD_KEYS = ("lmax", "zmin", "zmax", "dx")


def _cache_path_for_sim(cache_dir, sim_num) -> _Path:
    return _Path(cache_dir) / f"camb_cls_sim{int(sim_num)}.pkl"


def _guard_cosmo_match(expected, found, sim_num, *, rtol: float = 1e-6, atol: float = 1e-9):
    """Raise if the cached cosmology does not match the (re-derived) expected cosmology.

    This protects the shared cache from silently pairing CAMB Cls computed for one cosmology
    with a different cosmology for the same sim_id (e.g. prior/seed/preset drift between variates).
    """
    mismatched = []
    for k in COSMO_GUARD_KEYS:
        if k not in expected:
            mismatched.append((k, "absent in re-derived cosmology"))
            continue
        if k not in found:
            mismatched.append((k, "absent in cached cosmology"))
            continue
        if not np.isclose(float(expected[k]), float(found[k]), rtol=rtol, atol=atol):
            mismatched.append((k, float(expected[k]), float(found[k])))
    if mismatched:
        raise ValueError(
            f"CAMB Cls cache cosmology mismatch for sim {sim_num}: {mismatched}. "
            f"The cached cosmology differs from the deterministically re-derived one — refusing "
            f"to reuse stale/foreign Cls. Delete the cache entry or fix the prior/base-seed/preset."
        )


def _guard_grid_match(expected_grid, found_grid, sim_num, *, rtol: float = 1e-9, atol: float = 0.0):
    mismatched = []
    if int(expected_grid["lmax"]) != int(found_grid.get("lmax", -1)):
        mismatched.append(("lmax", int(expected_grid["lmax"]), found_grid.get("lmax")))
    for k in ("zmin", "zmax", "dx"):
        if not np.isclose(float(expected_grid[k]), float(found_grid.get(k, np.nan)), rtol=rtol, atol=atol):
            mismatched.append((k, float(expected_grid[k]), found_grid.get(k)))
    if mismatched:
        raise ValueError(
            f"CAMB Cls cache grid mismatch for sim {sim_num}: {mismatched}. "
            f"Cached spectra were computed for a different (lmax, zmin, zmax, dx) grid."
        )


def _atomic_pickle_dump(path, obj):
    """Write `obj` to `path` atomically (write temp + fsync + os.replace) so a killed run can
    never leave a partially-written cache file that a later run would read as valid."""
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{_os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            _pickle.dump(obj, f, protocol=_pickle.HIGHEST_PROTOCOL)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, path)
    finally:
        if _os.path.exists(tmp):
            try:
                _os.remove(tmp)
            except Exception:
                pass


def compute_or_load_glass_cls(
    cosmo_params,
    grid,
    *,
    cache_dir,
    sim_num,
    camb_limits,
):
    """Return (shells, glass_cls) for `cosmo_params`, using a shared on-disk cache keyed by sim_id.

    - Cache hit: load `{cosmo_params, grid, shells, glass_cls}`, guard that the cached cosmology
      and grid match `cosmo_params`/`grid`, and return the cached spectra (skipping CAMB entirely).
    - Cache miss: run the existing CAMB subprocess for `cosmo_params`, then atomically persist the
      result to the cache so later runs / analysis variates can reuse it.

    `cosmo_params` should contain (at least) the COSMO_GUARD_KEYS in physical units; only the
    cosmology drives CAMB, so a_ia/b_ia are intentionally excluded from the cache and the worker.
    `grid` is a dict with keys lmax, zmin, zmax, dx.
    """
    cache_path = _cache_path_for_sim(cache_dir, sim_num)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                data = _pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read CAMB Cls cache {cache_path} for sim {sim_num}: {e}")
        _guard_cosmo_match(cosmo_params, data.get("cosmo_params", {}), sim_num)
        _guard_grid_match(grid, data.get("grid", {}), sim_num)
        return data["shells"], data["glass_cls"]

    # Miss: only the cosmology drives CAMB; pass the guard keys as the worker param_dict.
    worker_params = {k: cosmo_params[k] for k in COSMO_GUARD_KEYS if k in cosmo_params}
    npz_out_path = compute_camb_glass_in_child_npz_subproc(
        worker_params,
        grid["lmax"],
        grid["zmin"],
        grid["zmax"],
        grid["dx"],
        mem_limit_gb=camb_limits["mem_limit_gb"],
        timeout_s=camb_limits["timeout_s"],
        sim_tag=f"sim{sim_num}",
    )
    shells, glass_cls = load_camb_child_pickle(npz_out_path, remove_after_load=True)

    _atomic_pickle_dump(
        cache_path,
        {
            "cosmo_params": {k: float(cosmo_params[k]) for k in COSMO_GUARD_KEYS if k in cosmo_params},
            "grid": {
                "lmax": int(grid["lmax"]),
                "zmin": float(grid["zmin"]),
                "zmax": float(grid["zmax"]),
                "dx": float(grid["dx"]),
            },
            "shells": shells,
            "glass_cls": glass_cls,
        },
    )
    return shells, glass_cls
