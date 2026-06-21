# Hybrid training — remote deploy (quick launch)

Two-stage hybrid SBI on the cluster (**hypatia**, env `glass`, branch `kids-preparation`). Drive the
cluster ONLY via `python .claude/cluster/run_remote.py <verb>` (always `--dry-run` mutating verbs first).
Config lives in `config/kids_legacy.py`; checkpoints/data under `/share/gpu5/asaoulis/transfer_{models,datasets}`.

1. **Prebake the map store** (extract 1 E-variant N+S, downcast f16, → gpu5 l40s-local; ~4.5× faster than gpu4 NFS):
   `run_remote.py prebake --src-datasets-root gpu4 --src-dir <RAW_DS> --out-dir <STORE> --eb-variant <fwhmK_lminL_lcutC> --keep-variant-tag`
   Verify the tag is right: `logs --name prebake` → first progress line `ok>0` (=`E_<tag>` exists; `ok=0` ⇒ wrong tag).
   Then `data-ls --rel <STORE>` → file count ≈ source, real `du` (~120G).
2. **Register the experiment** in `config/kids_legacy.py` (copy `_hybrid(...)` / band dict; edit ONLY this file):
   set `data_patterns`→`/share/gpu5/.../<STORE>/output_*.h5`, `eb_map_variant`→`<fwmhK_lminL_lcutC>`,
   `pretrained_band_ckpt_path`→ the band ckpt dir, `freeze_band=True`, and `repeat_indices` (e.g. `[0,1]`/`[2,3]`
   for 2 members/sub). Band exp = `kids_bandpowers_mlp` on `mixed_bandpowers` (smoothing-independent → 1 band per repeat).
   NB: each repeat `i` → run folder `pretrain_ncosmoNone_{i}`; the hybrid auto-loads band `i` by match-string `_{i}`.
3. **Smoke (local gate):** `python .claude/cluster/smoke_test_experiment.py --experiment <name>` → finite loss.
   (`train` runs this hard gate itself; band-load is NOT provable locally — proven on the cluster log in step 6.)
4. **Commit + push + sync** (sync `git reset --hard`s the cluster to a *pushed* rev): `git push origin kids-preparation`
   then `run_remote.py sync` (verify printed HEAD == your commit). **Push must land before sync.**
   Push over **SSH** — in-session HTTPS has no creds; one-time set the origin push-URL:
   `git remote set-url --push origin git@github.com:asaoulis/glass_gower_transfer.git` (id_rsa is in the agent at `~/.ssh/agent.sock`; fetch stays HTTPS for the cluster).
5. **Train** (Stage I band first, wait for its ckpts; then Stage II hybrid). Force l40s for gpu5-locality:
   `run_remote.py train --exp <name> --gpu l40s --ncpu 16` (`--dry-run` first). `status` BEFORE every (re)submit —
   submits are NOT idempotent (a dropped ssh link can still create the job; never blindly retry).
   - **GPU preference**: auto-pick order is **l40s → a100 → v100** (l40s = gpu5-local/fast; a100/v100 read over NFS, ~4.5× slower). Omit `--gpu` to use the order, or force one.
   - **`--mem-gb N`**: lowers `#SBATCH --mem` (default 64G) to fit a RAM-contended node — pair it with fewer dataloader workers in the config (e.g. `num_workers=8, prefetch_factor=2`) so the smaller mem is OOM-safe. (Plumbed via the gatekeeper's `TRAIN_MEM`; needs a `bootstrap_install.sh` after any gatekeeper edit.)
6. **Verify band-load** (the gate) in the Stage-II log: `run_remote.py logs --name <hybrid_exp>` → for each repeat
   `Best checkpoints found: ['.../pretrain_ncosmoNone_{i}/...']` (non-empty) + `[load_partial_weights] KidsBandpowersMLP ... Loaded keys: 14` (N>0).
   `Loaded keys: 0` / empty list / `No weights were loaded!` ⇒ STOP (wrong band dir or match-string).
7. **Monitor:** `status` + `logs` each poll; l40s+gpu5-local ≈135 smp/s (≈30 ⇒ store not gpu5-local). PENDING(Priority/Resources) is fine — don't thrash resubmits.

Protected (never edit without explicit OK): `src/cosmology/`, `src/KiDS/{systematics,tomo}.py` (guard hook blocks them).
