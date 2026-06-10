import argparse
import csv
import os

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


DEFAULT_TAGS = ["reward/train", "success/train", "steps/train"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar metrics to CSV.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-csv", default="results/tensorboard_scalars.csv")
    parser.add_argument("--tag", action="append", help="Scalar tag to export. Defaults to reward/success/steps train.")
    parser.add_argument("--run-filter", action="append", help="Only export runs whose path contains this text.")
    parser.add_argument("--max-file-mb", type=float, default=100.0, help="Skip event files larger than this size. Use 0 for no limit.")
    parser.add_argument("--plot-dir", default=None, help="Optional directory for PNG plots if matplotlib is installed.")
    return parser.parse_args()


def iter_event_files(runs_dir, run_filters=None, max_file_mb=100.0):
    for root, _, files in os.walk(runs_dir):
        run_name = os.path.relpath(root, runs_dir)
        if run_filters and not any(filter_text in run_name for filter_text in run_filters):
            continue
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                size_mb = os.path.getsize(event_file) / (1024 * 1024)
                if max_file_mb > 0 and size_mb > max_file_mb:
                    print(f"Skipped {event_file} ({size_mb:.1f} MB > {max_file_mb:.1f} MB)")
                    continue
                yield root, event_file


def scalar_value(summary_value):
    if summary_value.HasField("simple_value"):
        return summary_value.simple_value

    tensor = summary_value.tensor
    if tensor.float_val:
        return tensor.float_val[0]
    if tensor.double_val:
        return tensor.double_val[0]
    if tensor.int_val:
        return tensor.int_val[0]
    return None


def export_scalars(runs_dir, tags, run_filters=None, max_file_mb=100.0):
    rows = []
    requested_tags = set(tags) if tags is not None else None

    for event_dir, event_file in iter_event_files(runs_dir, run_filters=run_filters, max_file_mb=max_file_mb):
        run_name = os.path.relpath(event_dir, runs_dir)
        loader = EventFileLoader(event_file)
        for event in loader.Load():
            if not event.summary:
                continue
            for value in event.summary.value:
                if requested_tags is not None and value.tag not in requested_tags:
                    continue
                scalar = scalar_value(value)
                if scalar is None:
                    continue
                rows.append(
                    {
                        "run": run_name,
                        "tag": value.tag,
                        "step": event.step,
                        "value": scalar,
                        "wall_time": event.wall_time,
                    }
                )
    return rows


def write_csv(path, rows):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "tag", "step", "value", "wall_time"])
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped PNG plots. CSV export is complete.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["run"], row["tag"]), []).append(row)

    for (run, tag), group in grouped.items():
        group = sorted(group, key=lambda row: row["step"])
        steps = [row["step"] for row in group]
        values = [row["value"] for row in group]
        safe_run = run.replace("\\", "_").replace("/", "_").replace(":", "_")
        safe_tag = tag.replace("/", "_").replace(":", "_")

        plt.figure(figsize=(8, 4))
        plt.plot(steps, values)
        plt.xlabel("Episode")
        plt.ylabel(tag)
        plt.title(f"{run} - {tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{safe_run}_{safe_tag}.png"))
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    tags = args.tag if args.tag else DEFAULT_TAGS
    rows = export_scalars(args.runs_dir, tags, run_filters=args.run_filter, max_file_mb=args.max_file_mb)
    write_csv(args.output_csv, rows)
    print(f"Exported {len(rows)} scalar rows to {args.output_csv}")

    if args.plot_dir:
        plot_rows(rows, args.plot_dir)
