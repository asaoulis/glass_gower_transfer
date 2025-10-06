# ---------------- Subprocess .npz wrapper with robust logging ----------------
import os
import sys
import tempfile
import shutil
import pickle
import subprocess
import time

import numpy as np

def compute_levin_glass_in_child_npz_subproc(
    param_dict,
    zb, z_grid, chi_grid, extended_k, extended_pk,
    mem_limit_gb: float = 50.0,
    timeout_s: int = 1800,
    sim_tag: str = None,
    tmpdir_base: str = "/share/gpu5/asaoulis/tmp",
    compress: bool = False,
):
    """
    Spawn a worker subprocess that recomputes CAMB and runs the heavy levin+glass calls,
    saving results into an .npz file.

    Returns: absolute path to the output .npz file on success.
    Raises: TimeoutError or RuntimeError (including child's traceback if available).
    """
    # create tempdir
    prefix = f"levin_npz_subproc_{sim_tag or int(time.time())}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=tmpdir_base)
    out_npz = os.path.join(tmpdir, "levin_child_outputs.npz")
    inputs_pkl = os.path.join(tmpdir, "inputs.pkl")
    worker_py = os.path.join(tmpdir, "levin_child_worker.py")
    stdout_log = os.path.join(tmpdir, "worker.stdout.log")
    stderr_log = os.path.join(tmpdir, "worker.stderr.log")
    errfile = out_npz + ".err"

    try:
        # 1) Serialize all inputs into inputs.pkl so the worker doesn't need argument pickling
        with open(inputs_pkl, "wb") as f:
            pickle.dump({
                "param_dict": param_dict,
                "zb": zb,
                "z_grid": z_grid,
                "chi_grid": chi_grid,
                "extended_k": extended_k,
                "extended_pk": extended_pk,
                "mem_limit_gb": mem_limit_gb,
                "compress": bool(compress)
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2) Write worker script into tmpdir
        worker_code = r'''
#!/usr/bin/env python3
import resource, pickle, sys, os, traceback, time
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
        zb = data["zb"]
        z_grid = data["z_grid"]
        chi_grid = data["chi_grid"]
        extended_k = data["extended_k"]
        extended_pk = data["extended_pk"]
        mem_limit_gb = float(data.get("mem_limit_gb", 30.0))
        compress = bool(data.get("compress", False))

        # enforce memory cap
        bytes_limit = int(mem_limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))

        # IMPORT heavy modules here
        import levin as _levin
        import glass_utils as _glass_utils
        import parameters as _parameters
        import camb as _camb

        # Rebuild CAMB state inside child
        cosmo, pars = _parameters.build_cosmology(param_dict)
        results = _camb.get_results(pars)
        results.calc_power_spectra(pars)

        # heavy C++ allocations
        ws_local, lp_local, ell_local = _levin.setup_levin_power(
            zb, z_grid, chi_grid, extended_k, extended_pk, results, pars
        )
        glass_cls_local, ws_final, n_glass_shells_local = _glass_utils.compute_glass_cls(
            lp_local, ws_local, ell_local
        )
        # Save everything via pickle
        with open(out_npz, "wb") as f:
            pickle.dump(
                {
                    "glass_cls": glass_cls_local,
                    "ws": ws_final,
                    "n_glass_shells": int(n_glass_shells_local),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        # force fsync directory to reduce race conditions
        try:
            d = os.path.dirname(out_npz) or "."
            fd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass

        # normal exit
        sys.exit(0)

    except MemoryError as me:
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

        # 3) Launch the worker subprocess (use same interpreter)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_python = os.path.join(conda_prefix, "bin", "python")
        else:
            conda_python = sys.executable  # fallback
        
        cmd = [conda_python, worker_py, inputs_pkl, out_npz]
        env = os.environ.copy()
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
                raise TimeoutError(f"Worker timed out after {timeout_s} s (tag={sim_tag}). See logs in {tmpdir}")

        # 4) Inspect exit code and produce helpful errors
        exitcode = proc.returncode
        if exitcode != 0:
            # If worker wrote .err with traceback, include it
            if os.path.exists(errfile):
                with open(errfile, "r") as ef:
                    child_tb = ef.read()
                raise RuntimeError(f"Worker failed (exitcode={exitcode}). Traceback from worker (.err):\n{child_tb}\n\nSee logs: {stdout_log}, {stderr_log}")
            else:
                # Read stderr to show what happened
                try:
                    with open(stderr_log, "r") as ef:
                        stderr_txt = ef.read()
                except Exception:
                    stderr_txt = "<could not read stderr log>"
                raise RuntimeError(f"Worker exited with code {exitcode}. Stderr:\n{stderr_txt}\nSee logs in {tmpdir}")

        # success: verify file exists
        if not os.path.exists(out_npz):
            raise RuntimeError(f"Worker returned exitcode 0 but output file missing at {out_npz}. See logs in {tmpdir}")

        # return path to output file (caller can load and then remove it)
        return os.path.abspath(out_npz)

    except Exception:
        # On failure, try to attach useful logs to the exception before re-raising
        # prefer to leave tmpdir in place for debugging
        raise

    # Note: do not auto-delete tmpdir so you can inspect worker logs on failure.
# ---------------- end wrapper ----------------
import os, pickle, shutil

def load_levin_child_pickle(out_path, remove_after_load: bool = True):
    """
    Load pickle file produced by compute_levin_glass_in_child_npz_subproc worker.
    Returns (glass_cls, ws, n_glass_shells).
    """
    try:
        with open(out_path, "rb") as f:
            data = pickle.load(f)
        glass_cls = data["glass_cls"]
        ws = data["ws"]
        n_glass_shells = data["n_glass_shells"]
    except Exception as e:
        raise RuntimeError(f"Failed to load child pickle {out_path}: {e}")

    if remove_after_load:
        try:
            tmpdir = os.path.dirname(out_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return glass_cls, ws, n_glass_shells
