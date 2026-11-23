"""Live loss plot for training runs.

Reads one or more train.py log files and draws the loss curve as it
grows, updating every 2 seconds. The per-step loss is noisy (each step
trains on a different random batch), so the plot shows the raw loss
faint and a 20-step moving average bold — the same last-20 average
used to read a run's final loss.

The legend shows each run's current 20-step averaged loss next to its
name, updating live as the run progresses.

Usage:
    python3 benchmarks/plot_loss.py train-d256-n2-bpe.log
    python3 benchmarks/plot_loss.py train-d256-n2-bpe.log train-d256-n2-bpe-wikitext.log

Pass several logs to compare runs on one plot (each gets its own
colour and a legend entry from the log filename).
"""

import argparse
import os
import re
import sys
import time

import matplotlib.pyplot as plt

LOSS_RE = re.compile(r"step\s+(\d+)\s+loss\s+([0-9.]+)")
WINDOW = 20  # moving-average window, matches the last-20 loss reading


def read_series(path):
    """Return (steps, losses) parsed from a train.py log file."""
    steps, losses = [], []
    try:
        with open(path) as f:
            for line in f:
                m = LOSS_RE.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
    except FileNotFoundError:
        pass
    return steps, losses


def moving_average(values, window=WINDOW):
    """Trailing moving average; the first points average what's there."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i + 1 - lo))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="train.py log files to plot")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between refreshes (default 2)",
    )
    args = parser.parse_args()

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("training loss (live)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    # One legend entry per run (the bold 20-step average line). The raw
    # line is plotted faint with no legend entry, to keep the legend
    # readable when many runs are compared. The entry's label carries
    # the run's current averaged loss and is updated every refresh.
    lines = {}
    names = {}
    for path in args.logs:
        label = os.path.basename(path).replace("train-", "").replace(".log", "")
        (raw,) = ax.plot([], [], alpha=0.25, label="_nolegend_")
        (avg,) = ax.plot([], [], linewidth=2, label=label)
        lines[path] = (raw, avg)
        names[path] = label
    ax.legend(loc="upper right")

    print("plotting", ", ".join(args.logs), "— close the window to stop")
    try:
        while True:
            for path, (raw, avg) in lines.items():
                steps, losses = read_series(path)
                if not steps:
                    continue
                raw.set_data(steps, losses)
                avgs = moving_average(losses)
                avg.set_data(steps, avgs)
                # Update the legend entry with the current averaged loss.
                avg.set_label(f"{names[path]} — {avgs[-1]:.3f}")
            ax.legend(loc="upper right")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print("stopped")


if __name__ == "__main__":
    main()
