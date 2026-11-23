#!/usr/bin/env python3
"""Show progress + ETA for all training runs in this directory.
Run from sprint-03-transformer/:

    python3 check_progress.py              # print once and exit
    python3 check_progress.py --watch      # refresh every 30s until Ctrl+C
    python3 check_progress.py --watch 10   # refresh every 10s

Reads each train-*.log, finds the total steps (from the header) and the
latest step/elapsed time, and prints progress % and ETA.
"""

import argparse
import datetime
import glob
import os
import re
import sys
import time


def parse_log(path):
    total_steps = None
    last_step = None
    last_elapsed = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"epochs [\d.]+ -> (\d+) steps", line)
            if m:
                total_steps = int(m.group(1))
            m2 = re.search(r"engine numpy.*steps (\d+),", line)
            if m2 and total_steps is None:
                total_steps = int(m2.group(1))
            m3 = re.match(r"step\s+(\d+)\s+loss\s+[\d.]+\s+\((\d+)s\)", line)
            if m3:
                last_step = int(m3.group(1))
                last_elapsed = int(m3.group(2))
    return total_steps, last_step, last_elapsed


def fmt_dur(seconds):
    if seconds is None or seconds < 0:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def render():
    rows = []
    for path in sorted(glob.glob("train-*.log")):
        total, step, elapsed = parse_log(path)
        name = os.path.basename(path)
        if step is None:
            rows.append((name, "-", "-", "-", "-"))
            continue
        if total:
            pct = 100.0 * step / total
            if step > 0 and elapsed is not None:
                per_step = elapsed / max(step, 1)
                eta = fmt_dur((total - step) * per_step)
            else:
                eta = "-"
            rows.append(
                (name, f"{step:,}/{total:,}", f"{pct:.1f}%", fmt_dur(elapsed), eta)
            )
        else:
            rows.append((name, f"{step:,}/?", "-", fmt_dur(elapsed), "-"))
    lines = []
    lines.append(f"{'log':<42} {'step':<18} {'done':<7} {'elapsed':<9} {'ETA':<8}")
    for r in rows:
        lines.append(f"{r[0]:<42} {r[1]:<18} {r[2]:<7} {r[3]:<9} {r[4]:<8}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show training progress + ETA.")
    parser.add_argument(
        "--watch",
        nargs="?",
        const=30,
        type=int,
        default=None,
        metavar="SECONDS",
        help="refresh every N seconds (default 30) until Ctrl+C",
    )
    args = parser.parse_args()

    if args.watch is None:
        print(render())
        return

    interval = args.watch
    first = True
    try:
        while True:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            header = f"Training progress — updated {now} (refreshing every {interval}s, Ctrl+C to stop)\n"
            body = render()
            if first:
                print(header)
                print(body)
                first = False
            else:
                # Move the cursor home and overwrite in place instead of
                # clearing the whole screen, so there's no blank flash.
                # The trailing clear wipes any leftover lines if the table
                # got shorter since the last refresh.
                print("\033[H" + header + "\n" + body + "\033[J", end="")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
