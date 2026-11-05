"""Feed queued OT-loss benchmark shards to Slurm within this thread's slot share.

queue.tsv rows: tag, deploy dir, config, time limit, shard index (priority order).
Each poll sets this thread's share (fits plus this feeder) to
    min(SHARE, PROJECT_CAP - noise-loss jobs - (dc-fill tasks + 1)),
with noise-loss counted as at least NOISE_SHARE, so the share grows only as the
readiness feeder drains. Fit jobs are named dc-ot-<tag>; pending ones count too.
Shards whose events.csv already reports completed/skipped_cap are skipped.
When the queue is drained and no fit is left, it submits one collector per config.
Before its own time limit it resubmits itself (afterany on this job) and exits.
"""
import csv
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARE = int(os.environ.get("FEEDER_SHARE", "10"))
PROJECT_CAP = int(os.environ.get("FEEDER_PROJECT_CAP", "80"))
NOISE_SHARE = int(os.environ.get("FEEDER_NOISE_SHARE", "10"))
LIFETIME = float(os.environ.get("FEEDER_SECONDS", str(4 * 86400 + 20 * 3600)))
POLL = int(os.environ.get("FEEDER_POLL", "300"))
USER = os.environ.get("USER", "dmillard")
FEEDER_NAME = "dc-ot-feeder"


def log(message):
    with (HERE / "feeder.log").open("a") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def tasks(job_id):
    """Count tasks in a squeue id such as 123, 123_4, or 123_[0-16:2,20%5]."""
    if "[" not in job_id:
        return 1
    total = 0
    for part in job_id.split("[")[1].split("]")[0].split("%")[0].split(","):
        bounds, _, step = part.partition(":")
        first, _, last = bounds.partition("-")
        total += (int(last or first) - int(first)) // int(step or 1) + 1
    return total


def usage():
    """Return (fit jobs of ours, free fit slots), or None if squeue failed."""
    # squeue -r drops the last task of a stepped pending array, so expand ids here.
    out = subprocess.run(["squeue", "-u", USER, "-h", "-o", "%2000i|%j"], capture_output=True,
                         text=True, env={**os.environ, "SLURM_BITSTR_LEN": "0"})
    if out.returncode:  # transient: submit nothing this poll
        log(f"squeue failed: {out.stderr.strip()}")
        return None
    jobs = [line.strip().split("|", 1) for line in out.stdout.splitlines()]
    ours = sum(tasks(i) for i, n in jobs if n.startswith("dc-ot-") and n != FEEDER_NAME)
    fill = sum(tasks(i) for i, n in jobs if n.startswith("dc-fill-"))
    noise = sum(tasks(i) for i, n in jobs if n.startswith(("dc-nl", "diffclarinet-")))
    share = min(SHARE, PROJECT_CAP - max(NOISE_SHARE, noise) - (fill + 1))
    return ours, share - 1 - ours


def output_dir(deploy, config):
    for line in (Path(deploy) / config).read_text().splitlines():
        if line.startswith("output_dir:"):
            return Path(deploy) / line.split(":", 1)[1].strip()
    raise ValueError(config)


def finished(deploy, config, shard):
    events = output_dir(deploy, config) / "shards" / f"{shard:05d}" / "events.csv"
    if not events.exists():
        return False
    lines = events.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    latest = {row["run_id"]: row["status"] for row in csv.DictReader(lines)}
    return bool(latest) and all(s in {"completed", "skipped_cap"} for s in latest.values())


def sbatch(args):
    out = subprocess.run(["sbatch", "--parsable", *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip().split(";")[0]


def submit(tag, deploy, config, limit, shards):
    base = min(shards)
    ids = ",".join(str(s - base) for s in shards)
    python = f"{deploy}/.venv/bin/python"
    job = sbatch([f"--array={ids}", f"--time={limit}", f"--job-name=dc-ot-{tag}",
                  f"--chdir={deploy}", f"--output={deploy}/slurm_logs/ot_{tag}_%A_%a.log",
                  f"--export=ALL,BENCHMARK_PYTHON={python},BENCHMARK_CONFIG={config},"
                  f"BENCHMARK_SHARD_SIZE=1,BENCHMARK_SHARD_STRIDE=1,BENCHMARK_SHARD_OFFSET={base}",
                  f"{deploy}/slurm/delay_benchmark_strided.sbatch"])
    with (HERE / "submitted.tsv").open("a") as stream:
        stream.write(f"{job}\t{tag}\t{config}\t{base}\t{' '.join(map(str, shards))}\n")
    log(f"submitted {job} {tag} {len(shards)} shards (base {base})")
    return job


def main():
    def read_queue():
        return [line.split("\t") for line in (HERE / "queue.tsv").read_text().splitlines() if line]

    queue = read_queue()
    pointer_file = HERE / "pointer"
    pointer = int(pointer_file.read_text()) if pointer_file.exists() else 0
    start = time.time()
    log(f"start job {os.environ.get('SLURM_JOB_ID')} pointer {pointer}/{len(queue)} share {SHARE}")
    while True:
        # Rows may be appended (or unconsumed rows reordered) while this runs.
        queue = read_queue()
        state = usage()
        free = state[1] if state is not None else 0
        while free > 0 and pointer < len(queue):
            before = pointer
            tag, deploy, config, limit, _ = queue[pointer]
            batch = []
            while (pointer < len(queue) and len(batch) < free
                   and queue[pointer][:4] == [tag, deploy, config, limit]):
                shard = int(queue[pointer][4])
                if batch and max(batch + [shard]) - min(batch + [shard]) > 1000:
                    break
                pointer += 1
                if not finished(deploy, config, shard):
                    batch.append(shard)
            if batch:
                try:
                    submit(tag, deploy, config, limit, batch)
                except RuntimeError as error:  # e.g. submit limit; retry next poll
                    log(f"sbatch failed, will retry: {error}")
                    pointer = before
                    break
                free -= len(batch)
            pointer_file.write_text(str(pointer))
        state = usage()
        if pointer >= len(queue) and state is not None and state[0] == 0:
            done = HERE / "collectors_submitted"
            if not done.exists():
                for deploy, config in dict.fromkeys((r[1], r[2]) for r in queue):
                    job = sbatch([f"--chdir={deploy}", "--job-name=dc-ot-collect",
                                  f"--output={deploy}/slurm_logs/collect_ot_%j.log",
                                  f"--export=ALL,BENCHMARK_PYTHON={deploy}/.venv/bin/python,"
                                  f"BENCHMARK_CONFIG={config},BENCHMARK_SHARD_SIZE=1",
                                  f"{deploy}/slurm/delay_benchmark_collect.sbatch"])
                    log(f"collector {job} {deploy} {config}")
                done.write_text("done\n")
            log("queue drained; exiting")
            return
        if time.time() - start > LIFETIME:
            job = sbatch([f"--dependency=afterany:{os.environ['SLURM_JOB_ID']}",
                          str(HERE / "ot_feeder.sbatch")])
            log(f"lifetime reached; successor {job}")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # leave a trace; the queue state is in pointer/submitted.tsv
        log(f"ERROR {type(error).__name__}: {error}")
        raise
