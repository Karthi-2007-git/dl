"""Find a close local experiment title and print its code."""
import argparse
from pathlib import Path
from dl_lab_remote import EXPERIMENTS, choose

p = argparse.ArgumentParser()
p.add_argument("question", nargs="*")
p.add_argument("--list", action="store_true")
a = p.parse_args()

if a.list:
    print("\n".join(f"{x} -> {y}" for x, y, _ in EXPERIMENTS))
else:
    (title, file, _), score = choose(" ".join(a.question))
    print(f"# {title} | estimate {score:.2f}\n")
    print(Path(file).read_text(encoding="utf8"))
