#!/usr/bin/env python3
"""
crimm High-Throughput Pipeline Benchmark
=========================================
Tests randomly selected PDB structures through the crimm structure preparation
pipeline and reports per-stage success/failure rates.

Pipeline stages:
  1. fetch       - Download mmCIF from RCSB and parse into a Structure
  2. organize    - Build OrganizedModel (classify chains by type)
  3. loop_build  - Fill missing residues using AlphaFold templates
  4. topo_gen    - Generate CHARMM force field topology
  5. charmm_load - Load into pyCHARMM (skipped if pyCHARMM is unavailable)

Each structure is processed in a dedicated subprocess spawned with the 'spawn'
start method, so a hard pyCHARMM crash (segfault / kernel abort) only kills
the worker and the main process survives.  The worker writes a JSON checkpoint
after every stage, so partial results are always recoverable.

Output:
  Results are appended to a CSV file after each structure so the run can be
  resumed safely.  A summary is printed at the end (and on SIGINT/SIGTERM).

Usage:
  python benchmark_pipeline.py                        # 5000 structures, default seed
  python benchmark_pipeline.py --n 200                # run only 200 structures
  python benchmark_pipeline.py --seed 7               # different random sample
  python benchmark_pipeline.py --ids ids.txt          # use explicit list of PDB IDs
  python benchmark_pipeline.py --out results.csv      # custom output file
  python benchmark_pipeline.py --resume               # skip IDs already in output CSV
  python benchmark_pipeline.py --no-charmm            # skip pyCHARMM loading stage
  python benchmark_pipeline.py --cgenff /path/cgenff  # enable CGenFF ligand support
  python benchmark_pipeline.py --timeout 180          # per-structure timeout (seconds)
"""

import csv
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import tempfile
import threading
import time
import traceback
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields as dc_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STAGES = ("fetch", "organize", "loop_build", "topo_gen", "charmm_load")
STATUS_OK      = "ok"
STATUS_FAIL    = "fail"
STATUS_SKIP    = "skip"
STATUS_TIMEOUT = "timeout"
STATUS_PARTIAL = "partial"   # loop_build: some gaps repaired but not all
STATUS_CRASH   = "crash"     # subprocess killed by signal (hard crash)

RCSB_SEARCH_URL  = "https://search.rcsb.org/rcsbsearch/v1/query"
RCSB_HOLDINGS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"

CSV_FIELDNAMES = [
    "pdb_id",
    "fetch", "organize", "loop_build", "topo_gen", "charmm_load",
    "fetch_err", "organize_err", "loop_build_err", "topo_gen_err", "charmm_load_err",
    "n_protein", "n_gaps_found", "n_gaps_repaired",
    "elapsed_s", "timestamp",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Result:
    pdb_id: str
    fetch: str = STATUS_SKIP
    organize: str = STATUS_SKIP
    loop_build: str = STATUS_SKIP
    topo_gen: str = STATUS_SKIP
    charmm_load: str = STATUS_SKIP
    fetch_err: str = ""
    organize_err: str = ""
    loop_build_err: str = ""
    topo_gen_err: str = ""
    charmm_load_err: str = ""
    n_protein: int = 0
    n_gaps_found: int = 0
    n_gaps_repaired: int = 0
    elapsed_s: float = 0.0
    timestamp: str = ""

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ── PDB ID retrieval ──────────────────────────────────────────────────────────

def _fetch_protein_ids_from_search() -> list[str]:
    """Query RCSB search API for entries that contain at least one L-polypeptide."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "negation": False,
                "value": "Polypeptide(L)",
            },
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],
        },
    }
    log.info("Querying RCSB search API for protein entries …")
    resp = requests.post(RCSB_SEARCH_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    ids = [hit["identifier"] for hit in data.get("result_set", [])]
    log.info("RCSB returned %d protein-containing entries", len(ids))
    return ids


def _fetch_all_entry_ids() -> list[str]:
    """Fall-back: fetch the full RCSB holdings list."""
    log.info("Fetching full RCSB holdings list …")
    resp = requests.get(RCSB_HOLDINGS_URL, timeout=120)
    resp.raise_for_status()
    ids = resp.json()
    log.info("Retrieved %d total RCSB entries", len(ids))
    return ids


def get_pdb_ids(n: int, seed: int) -> list[str]:
    """Return a reproducibly random sample of *n* protein PDB IDs."""
    try:
        all_ids = _fetch_protein_ids_from_search()
    except Exception as exc:
        log.warning("Search API failed (%s); falling back to full holdings", exc)
        all_ids = _fetch_all_entry_ids()

    rng = random.Random(seed)
    if len(all_ids) <= n:
        return all_ids
    return rng.sample(all_ids, n)


def load_ids_from_file(path: str) -> list[str]:
    """Read one PDB ID per line from a text file (strips whitespace, skips blanks)."""
    ids = []
    with open(path) as fh:
        for line in fh:
            pid = line.strip().upper()
            if pid and not pid.startswith("#"):
                ids.append(pid)
    return ids


def load_done_ids(csv_path: str) -> set[str]:
    """Return the set of PDB IDs already recorded in an existing results CSV."""
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add(row["pdb_id"].upper())
    return done


# ── Worker (runs in a dedicated subprocess) ───────────────────────────────────

def _short_err(exc: Exception) -> str:
    """One-line summary of an exception suitable for a CSV cell."""
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def _run_stages(pdb_id: str, cgenff_path: Optional[str], run_charmm: bool,
                result_path: str) -> None:
    """
    Worker entry point.  Runs inside a freshly spawned subprocess.

    All five pipeline stages are executed sequentially.  After every stage the
    current Result is serialised to *result_path* so that the parent process can
    recover partial results even if this process is killed by a hard crash
    (e.g., pyCHARMM segfault).  An atomic os.replace() is used so the parent
    never reads a half-written file.

    Parameters
    ----------
    pdb_id      : Four-character PDB accession code.
    cgenff_path : Path to the CGenFF executable, or None to skip ligand topo.
    run_charmm  : Whether to attempt the charmm_load stage.
    result_path : Path to the JSON checkpoint file shared with the parent.
    """
    # These imports happen inside the spawned process, not the parent.
    import warnings
    warnings.filterwarnings("ignore")

    from crimm.Fetchers import fetch_rcsb
    from crimm.StructEntities.OrganizedModel import OrganizedModel
    from crimm.Modeller import TopologyGenerator
    from crimm.Modeller.LoopBuilder import ChainLoopBuilder

    res = Result(pdb_id=pdb_id, timestamp=datetime.now(timezone.utc).isoformat())
    t0 = time.perf_counter()

    def _checkpoint():
        """Atomically write the current result state to disk."""
        res.elapsed_s = round(time.perf_counter() - t0, 2)
        tmp = result_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res.as_dict(), f)
        os.replace(tmp, result_path)

    # Create a fresh TopologyGenerator in this process.
    # RTF/PRM files are cached inside this object for the duration of the call.
    topo_gen = TopologyGenerator(cgenff_excutable_path=cgenff_path)

    # ── Stage 1: fetch ────────────────────────────────────────────────────────
    raw_model = None
    try:
        raw_model = fetch_rcsb(
            pdb_id,
            include_solvent=True,
            use_bio_assembly=True,
            organize=False,
        )
        res.fetch = STATUS_OK
    except Exception as exc:
        res.fetch = STATUS_FAIL
        res.fetch_err = _short_err(exc)
        _checkpoint()
        return
    _checkpoint()

    # ── Stage 2: organize ─────────────────────────────────────────────────────
    model = None
    try:
        model = OrganizedModel(raw_model)
        res.n_protein = len(model.protein)
        res.organize = STATUS_OK
    except Exception as exc:
        res.organize = STATUS_FAIL
        res.organize_err = _short_err(exc)
        _checkpoint()
        return
    _checkpoint()

    if not (model.protein or model.RNA or model.DNA):
        # Nothing more to do; remaining stages stay as STATUS_SKIP.
        return

    # ── Stage 3: loop_build ───────────────────────────────────────────────────
    gaps_found = 0
    gaps_repaired = 0
    loop_build_failed = False
    loop_build_err_msg = ""

    if not model.protein:
        res.loop_build = STATUS_SKIP
    else:
        for chain in model.protein:
            if chain.is_continuous():
                continue
            gaps_found += 1
            try:
                looper = ChainLoopBuilder(chain, inplace=True)
                looper.build_from_alphafold(include_terminal=False)
                if chain.is_continuous():
                    gaps_repaired += 1
            except Exception as exc:
                loop_build_failed = True
                loop_build_err_msg = _short_err(exc)

        res.n_gaps_found = gaps_found
        res.n_gaps_repaired = gaps_repaired

        if gaps_found == 0:
            res.loop_build = STATUS_OK
        elif loop_build_failed and gaps_repaired == 0:
            res.loop_build = STATUS_FAIL
            res.loop_build_err = loop_build_err_msg
            _checkpoint()
            return
        elif loop_build_failed or gaps_repaired < gaps_found:
            res.loop_build = STATUS_PARTIAL
        else:
            res.loop_build = STATUS_OK
    _checkpoint()

    # ── Stage 4: topo_gen ─────────────────────────────────────────────────────
    try:
        topo_gen.generate_model(model, coerce=True, QUIET=True)
        res.topo_gen = STATUS_OK
    except Exception as exc:
        res.topo_gen = STATUS_FAIL
        res.topo_gen_err = _short_err(exc)
        _checkpoint()
        return
    _checkpoint()

    # ── Stage 5: charmm_load ──────────────────────────────────────────────────
    # A checkpoint is written BEFORE this stage so that if pyCHARMM kills the
    # process mid-load, the parent can see stages 1–4 already succeeded.
    if run_charmm:
        try:
            from crimm.Adaptors.pyCHARMMAdaptors import load_model, empty_charmm
            empty_charmm()
            load_model(model)
            res.charmm_load = STATUS_OK
        except Exception as exc:
            res.charmm_load = STATUS_FAIL
            res.charmm_load_err = _short_err(exc)
            try:
                empty_charmm()
            except Exception:
                pass
        _checkpoint()


def _dispatch_worker(
    pdb_id: str,
    cgenff_path: Optional[str],
    run_charmm: bool,
    timeout: int,
) -> Result:
    """
    Spawn a fresh subprocess to process one PDB structure and return its Result.

    Uses the 'spawn' start method so the child has a completely clean Python
    environment (no inherited state from the parent, no pyCHARMM globals).

    The worker writes JSON checkpoints after each stage.  If it is killed or
    times out, whatever was last checkpointed is read back and the first
    unfinished stage is annotated with STATUS_CRASH or STATUS_TIMEOUT.
    """
    # Write an empty placeholder so the file always exists before the worker runs.
    placeholder = Result(
        pdb_id=pdb_id, timestamp=datetime.now(timezone.utc).isoformat()
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        result_path = f.name
        json.dump(placeholder.as_dict(), f)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_run_stages,
        args=(pdb_id, cgenff_path, run_charmm, result_path),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout if timeout > 0 else None)

    timed_out = proc.is_alive()
    if timed_out:
        proc.kill()
        proc.join()

    exitcode = proc.exitcode  # 0=clean  <0=killed by signal  >0=sys.exit(N)

    # Read whatever was last checkpointed by the worker.
    try:
        with open(result_path) as f:
            data = json.load(f)
        # Reconstruct Result, falling back to placeholder defaults for any
        # missing keys (forward-compatibility guard).
        defaults = placeholder.as_dict()
        defaults.update(data)
        result = Result(**{k: defaults[k] for k in defaults if k in
                           {field.name for field in dc_fields(Result)}})
    except Exception:
        result = placeholder
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    # Annotate the first stage that didn't complete with the failure reason.
    if timed_out or (exitcode != 0 and exitcode is not None):
        status = STATUS_TIMEOUT if timed_out else STATUS_CRASH
        label = (
            f"Process timed out after {timeout}s"
            if timed_out
            else f"Process killed by signal (exit code {exitcode})"
        )
        for stage in STAGES:
            if getattr(result, stage) == STATUS_SKIP:
                setattr(result, stage, status)
                setattr(result, f"{stage}_err", label)
                break

    return result


# ── CSV helpers ───────────────────────────────────────────────────────────────

def open_csv(csv_path: str, resume: bool):
    """Return an open CSV file handle and DictWriter; write header if new."""
    path = Path(csv_path)
    write_header = not path.exists() or not resume
    fh = open(csv_path, "a", newline="", buffering=1)   # line-buffered
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        fh.flush()
    return fh, writer


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(results: list[Result]):
    total = len(results)
    if total == 0:
        print("\nNo results to summarise.")
        return

    all_statuses = (STATUS_OK, STATUS_PARTIAL, STATUS_FAIL, STATUS_TIMEOUT,
                    STATUS_CRASH, STATUS_SKIP)

    print("\n" + "=" * 72)
    print(f"  crimm Pipeline Benchmark — Summary ({total} structures)")
    print("=" * 72)

    for stage in STAGES:
        counts: dict[str, int] = {s: 0 for s in all_statuses}
        for r in results:
            counts[getattr(r, stage)] = counts.get(getattr(r, stage), 0) + 1

        attempted = total - counts[STATUS_SKIP]
        if attempted == 0:
            pct = "—"
        else:
            pct = f"{(counts[STATUS_OK] + counts[STATUS_PARTIAL]) / attempted * 100:.1f}%"

        print(
            f"  {stage:<14}"
            f"  ok={counts[STATUS_OK]:5d}"
            f"  partial={counts[STATUS_PARTIAL]:5d}"
            f"  fail={counts[STATUS_FAIL]:5d}"
            f"  crash={counts[STATUS_CRASH]:4d}"
            f"  timeout={counts[STATUS_TIMEOUT]:4d}"
            f"  skip={counts[STATUS_SKIP]:5d}"
            f"  success={pct}"
        )

    # Loop-build gap stats
    total_gaps = sum(r.n_gaps_found for r in results)
    if total_gaps > 0:
        with_gaps = sum(1 for r in results if r.n_gaps_found > 0)
        repaired  = sum(r.n_gaps_repaired for r in results)
        print(
            f"\n  Loop build:  {with_gaps} structures had gaps  |  "
            f"{repaired}/{total_gaps} gaps repaired  "
            f"({repaired / total_gaps * 100:.1f}%)"
        )

    elapsed_vals = [r.elapsed_s for r in results if r.elapsed_s > 0]
    if elapsed_vals:
        print(
            f"\n  Elapsed per structure:  "
            f"mean={sum(elapsed_vals)/len(elapsed_vals):.1f}s  "
            f"min={min(elapsed_vals):.1f}s  "
            f"max={max(elapsed_vals):.1f}s"
        )

    print("=" * 72 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark crimm pipeline on randomly selected PDB structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=5000,
                    help="Number of structures to test.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for PDB ID sampling.")
    ap.add_argument("--ids", metavar="FILE",
                    help="Text file with explicit PDB IDs (one per line). "
                         "Overrides --n and --seed.")
    ap.add_argument("--out", default="benchmark_results.csv",
                    help="Output CSV file path.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip PDB IDs already present in --out.")
    ap.add_argument("--no-charmm", dest="no_charmm", action="store_true",
                    help="Skip the pyCHARMM loading stage entirely.")
    ap.add_argument("--cgenff", metavar="PATH", default=None,
                    help="Path to the CGenFF executable (enables ligand topology).")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-structure timeout in seconds (0 = disabled).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of structures to process in parallel. "
                         "Each worker runs in its own subprocess, so "
                         "pyCHARMM crashes are isolated per structure.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show DEBUG-level log messages.")
    return ap.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Determine PDB IDs to test ─────────────────────────────────────────────
    if args.ids:
        pdb_ids = load_ids_from_file(args.ids)
        log.info("Loaded %d PDB IDs from %s", len(pdb_ids), args.ids)
    else:
        pdb_ids = get_pdb_ids(args.n, args.seed)

    if args.resume:
        done = load_done_ids(args.out)
        before = len(pdb_ids)
        pdb_ids = [pid for pid in pdb_ids if pid.upper() not in done]
        log.info(
            "Resuming: skipped %d already-done structures; %d remaining",
            before - len(pdb_ids), len(pdb_ids),
        )

    if not pdb_ids:
        log.info("Nothing to do.")
        return

    run_charmm = not args.no_charmm
    log.info(
        "Will process %d structures → %s  "
        "(workers=%d, charmm_load=%s, timeout=%ds)",
        len(pdb_ids), args.out, args.workers, run_charmm, args.timeout,
    )

    # ── Set up CSV output ─────────────────────────────────────────────────────
    csv_fh, csv_writer = open_csv(args.out, resume=args.resume)

    # ── Shared state (accessed from worker threads) ───────────────────────────
    results: list[Result] = []
    # Lock serialises CSV writes, stage_counts updates, and pbar postfix refreshes.
    _lock = threading.Lock()

    # ── Graceful interrupt handler ────────────────────────────────────────────
    def _handle_signal(*_):
        log.info("Interrupted — printing partial summary …")
        with _lock:
            print_summary(results)
        csv_fh.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Live per-stage counters for tqdm postfix ──────────────────────────────
    COUNTED = (STATUS_OK, STATUS_PARTIAL, STATUS_FAIL, STATUS_CRASH, STATUS_TIMEOUT)
    stage_counts: dict[str, dict[str, int]] = {
        s: {st: 0 for st in COUNTED} for s in STAGES
    }

    def _postfix() -> dict:
        """Compact per-stage ok/fail counts for the tqdm bar (call under _lock)."""
        out = {}
        for s in STAGES:
            c = stage_counts[s]
            if not any(c.values()):
                continue
            val = f"ok={c[STATUS_OK]}"
            if c[STATUS_FAIL]:
                val += f" fail={c[STATUS_FAIL]}"
            if c[STATUS_PARTIAL]:
                val += f" ~={c[STATUS_PARTIAL]}"
            if c[STATUS_CRASH]:
                val += f" crash={c[STATUS_CRASH]}"
            if c[STATUS_TIMEOUT]:
                val += f" T/O={c[STATUS_TIMEOUT]}"
            out[s] = val
        return out

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Each thread calls _dispatch_worker(), which itself spawns a subprocess and
    # blocks until it finishes.  The GIL is released during proc.join(), so
    # N threads give true N-way parallelism over the subprocess work.
    with logging_redirect_tqdm():
        pbar = tqdm(
            total=len(pdb_ids),
            desc=f"benchmark ({args.workers}w)",
            unit="struct",
            dynamic_ncols=True,
            colour="cyan",
        )

        def _process_one(pdb_id: str) -> Result:
            """Called from a thread-pool thread."""
            pdb_id = pdb_id.upper()
            log.info("Dispatching worker for %s …", pdb_id)
            try:
                return _dispatch_worker(pdb_id, args.cgenff, run_charmm, args.timeout)
            except Exception as exc:
                log.error("[%s] dispatch error: %s", pdb_id, exc)
                log.debug(traceback.format_exc())
                return Result(
                    pdb_id=pdb_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    fetch=STATUS_FAIL,
                    fetch_err=f"Dispatch error: {_short_err(exc)}",
                )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {
                executor.submit(_process_one, pid): pid.upper()
                for pid in pdb_ids
            }
            for future in as_completed(future_to_id):
                pdb_id = future_to_id[future]
                result = future.result()   # never raises; _process_one catches all

                with _lock:
                    for stage in STAGES:
                        s = getattr(result, stage)
                        if s in COUNTED:
                            stage_counts[stage][s] += 1
                    pbar.set_postfix(_postfix())
                    pbar.update(1)
                    results.append(result)
                    csv_writer.writerow(result.as_dict())
                    csv_fh.flush()

                log.info(
                    "[%s] %.1fs — %s",
                    pdb_id,
                    result.elapsed_s,
                    "  ".join(f"{s}={getattr(result, s)}" for s in STAGES),
                )

        pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    csv_fh.close()
    print_summary(results)
    log.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
