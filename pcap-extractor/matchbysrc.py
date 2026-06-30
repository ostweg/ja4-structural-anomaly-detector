import ast
import csv
from collections import defaultdict
from pathlib import Path


def load_ja4h_by_socket(path):
    """Read JA4H records and index them by a composite (src_ip, src_port) key."""
    path = Path(path)
    ja4h_by_socket = defaultdict(list)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue  # Skip malformed lines safely
                
            # Extract source IP and source port for the composite key
            src_ip = str(record.get("src", ""))
            src_port = str(record.get("srcport", ""))
            socket_key = (src_ip, src_port)
            
            ja4h_by_socket[socket_key].append({
                "src": src_ip,
                "srcport": src_port,
                "dst_ip": record.get("dst"),
                "dst_port": record.get("dstport"),
                "ja4h": record.get("JA4H"),
            })
    return ja4h_by_socket


def join_ja4_to_ja4h_csv(ja4_path, ja4h_path, output_path):
    """Join JA4 rows to JA4H by (src_ip, src_port) and write matching rows to CSV."""
    ja4h_by_socket = load_ja4h_by_socket(ja4h_path)

    fieldnames = ["ja4", "ja4h", "type"]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with open(ja4_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                # Ensure the line has the expected number of tab-separated fields
                if len(parts) < 6:
                    continue
                
                #  standard JA4 CSV format mapping:
                ja4_src_ip = str(parts[0])    # Source IP
                ja4_src_port = str(parts[1])  # Source Port
                ja4 = parts[4]                # JA4 TLS fingerprint
                
                # Create the matching tuple key
                socket_key = (ja4_src_ip, ja4_src_port)

                # Look up and write matches
                for ja4h_record in ja4h_by_socket.get(socket_key, []):
                    if not ja4h_record.get("ja4h"):
                        continue
                    writer.writerow({
                        "ja4": ja4,
                        "ja4h": ja4h_record["ja4h"],
                        "type": "htbot",
                    })


def main():
    import argparse

    #code has been changed to execute batch processing of multiple pcaps

    parser = argparse.ArgumentParser(
        description="Join a JA4 text file and a JA4H file by source socket and write a CSV output."
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
        print("Creating joined CSV from JA4 and JA4H records via Socket Matching...")
        join_ja4_to_ja4h_csv(
            ja4_path=root / "kongtuke2" / "ja4_output.txt",
            ja4h_path=root / "kongtuke2" / "ja4h_output.txt",
            output_path=root / "ja4_joined_by_src.csv",
        )


if __name__ == "__main__":
    main()
