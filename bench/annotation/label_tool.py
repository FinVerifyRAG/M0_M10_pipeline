"""CSV labeling and a small local page for supported / unsupported.

Double annotation: two CSV files with the same atom_id and different
annotator_id values. Cohen's kappa is computed from those columns.
"""
from __future__ import annotations

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List

from bench.annotation.agreement import cohen_kappa


FIELDS = [
    "atom_id", "doc_id", "atom_text", "source_text",
    "label", "annotator_id", "query_date",
]


def export_csv(atoms_jsonl: str, csv_path: str) -> int:
    rows = []
    with open(atoms_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "atom_id": row.get("atom_id", ""),
                "doc_id": row.get("doc_id", ""),
                "atom_text": row.get("text") or row.get("atom_text", ""),
                "source_text": row.get("source_text") or row.get("cited_text", ""),
                "label": row.get("label") or "",
                "annotator_id": row.get("annotator_id") or "",
                "query_date": row.get("query_date") or "",
            })
    dest = Path(csv_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def kappa_from_csvs(path_a: str, path_b: str) -> dict:
    a = {r["atom_id"]: r["label"] for r in load_csv(path_a) if r.get("label")}
    b = {r["atom_id"]: r["label"] for r in load_csv(path_b) if r.get("label")}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"kappa": None, "n": 0, "reason": "no overlapping labeled atoms"}
    labels_a = [a[i] for i in shared]
    labels_b = [b[i] for i in shared]
    return {
        "kappa": cohen_kappa(labels_a, labels_b),
        "n": len(shared),
        "annotator_a": load_csv(path_a)[0].get("annotator_id", ""),
        "annotator_b": load_csv(path_b)[0].get("annotator_id", ""),
    }


def serve_labels(csv_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Local page: atom text, cited source, and a supported/unsupported choice."""
    path = Path(csv_path)

    class Handler(BaseHTTPRequestHandler):
        def _rows(self):
            return load_csv(str(path)) if path.exists() else []

        def do_GET(self):
            rows = self._rows()
            body = ["<html><body><h1>RegGuard labels</h1>"]
            for row in rows:
                body.append(
                    f"<form method='POST'><p><b>{row.get('atom_id')}</b></p>"
                    f"<p>{row.get('atom_text','')}</p>"
                    f"<pre>{row.get('source_text','')}</pre>"
                    f"<input type='hidden' name='atom_id' value='{row.get('atom_id','')}'/>"
                    f"<input name='annotator_id' placeholder='annotator id' value='{row.get('annotator_id','')}'/>"
                    f"<button name='label' value='supported'>supported</button>"
                    f"<button name='label' value='unsupported'>unsupported</button></form><hr/>"
                )
            body.append("</body></html>")
            data = "\n".join(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            fields = dict(part.split("=", 1) for part in raw.split("&") if "=" in part)
            from urllib.parse import unquote_plus
            fields = {k: unquote_plus(v) for k, v in fields.items()}
            rows = self._rows()
            for row in rows:
                if row.get("atom_id") == fields.get("atom_id"):
                    row["label"] = fields.get("label", "")
                    row["annotator_id"] = fields.get("annotator_id", "")
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
