import ast
import csv
from collections import defaultdict
from pathlib import Path


def load_ja4h_by_stream(path):
    path = Path(path)
    """Read JA4H records and index them by stream."""
    ja4h_by_stream = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = ast.literal_eval(line)
            stream = str(record.get("stream", ""))
            ja4h_by_stream[stream].append({
                "src": record.get("src"),
                "dst_ip": record.get("dst"),
                "dst_port": record.get("dstport"),
                "ja4h": record.get("JA4H"),
                "stream": stream,
            })
    return ja4h_by_stream


def join_ja4_to_ja4h_csv(ja4_path, ja4h_path, output_path):
    """Join JA4 rows to JA4H by stream and write matching rows to CSV."""
    ja4h_by_stream = load_ja4h_by_stream(ja4h_path)

    fieldnames = ["ja4", "ja4h", "type"]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with open(ja4_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 6:
                    continue
                ja4 = parts[4]
                stream = str(parts[5])

                for ja4h_record in ja4h_by_stream.get(stream, []):
                    if not ja4h_record.get("ja4h"):
                        continue
                    writer.writerow({
                        "ja4": ja4,
                        "ja4h": ja4h_record["ja4h"],
                        "type": "htbot",
                    })


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Join a JA4 text file and a JA4H file by stream and write a CSV output."
    )
    parser.add_argument("--ja4", type=Path, help="Path to the JA4 output file")
    parser.add_argument("--ja4h", type=Path, help="Path to the JA4H output file")
    parser.add_argument("--output", type=Path, help="Path to the joined CSV output")
    args = parser.parse_args()

    if args.ja4 and args.ja4h and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Joining {args.ja4} and {args.ja4h} into {args.output}")
        join_ja4_to_ja4h_csv(args.ja4, args.ja4h, args.output)
    else:
        root = Path(__file__).resolve().parent
        print("Creating joined CSV from JA4 and JA4H records...")
        join_ja4_to_ja4h_csv(
            ja4_path=root / "kongtuke2" / "ja4_output.txt",
            ja4h_path=root / "kongtuke2" / "ja4h_output.txt",
            output_path=root / "ja4_joined_by_stream.csv",
        )


if __name__ == "__main__":
    main()
