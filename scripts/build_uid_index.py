#!/usr/bin/env python3
"""Build a byte-offset UID index for GLM main.jsonl.

Streaming 18 GB to find 10 rows is wasteful. This creates a one-time index:
  uid -> (file_path, byte_offset, line_length)
so future lookups can seek directly. Takes ~4 min to build, after that lookups
are O(1).

Output: outputs/glm_main_uid_index.json (maps uid -> [file, offset, length])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("uid_index")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
HUB = Path("/data/huggingface/hub")


def main() -> None:
    glm_main = sorted(
        (HUB / "datasets--Jackrong--GLM-5.1-Reasoning-1M-Cleaned" / "snapshots")
        .glob("*/main.jsonl"))
    if not glm_main:
        log.error("main.jsonl not found"); return
    fpath = glm_main[0]
    log.info("indexing %s ...", fpath)

    index: dict[str, list] = {}
    n = 0
    with fpath.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line)
                uid = str(rec.get("id", ""))
                if uid:
                    index[uid] = [str(fpath), offset, len(line)]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            n += 1
            if n % 100000 == 0:
                log.info("  %d rows indexed (%d UIDs)", n, len(index))

    out = OUT / "glm_main_uid_index.json"
    out.write_text(json.dumps(index))
    log.info("done: %d rows, %d UIDs -> %s (%.1f MB)", n, len(index), out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
