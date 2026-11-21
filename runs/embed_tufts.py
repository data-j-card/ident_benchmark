import sys
import time
import zipfile

import cv2
import numpy as np

from benchid.embed import Embedder, dump_versions
from benchid.paths import EMBEDDINGS, TUFTS
from benchid.sample import bank_ids

ZIPS = sorted(TUFTS.glob("*/TD_RGB_*.zip"))


def main() -> None:
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    dump_versions(EMBEDDINGS / "versions.json")
    bank = {str(i) for i in bank_ids()}
    embed = Embedder()
    for z in ZIPS:
        dest = EMBEDDINGS / f"{z.stem}.npz"
        if dest.exists():
            print(f"{z.name}: ya existe, se salta")
            continue
        with zipfile.ZipFile(z) as zf:
            names = [
                n for n in zf.namelist()
                if n.endswith(".jpg") and n.split("/")[0] in bank
            ]
            if not names:
                print(f"{z.name}: ninguna identidad que procesar, se salta")
                continue
            rows, t0 = [], time.time()
            for k, n in enumerate(names, 1):
                img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
                rows.append((n, embed(img)))
                if k % 50 == 0 or k == len(names):
                    print(f"{z.name}: {k}/{len(names)} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(
            dest,
            photo=np.array([n for n, _ in rows]),
            identity=np.array([int(n.split("/")[0]) for n, _ in rows]),
            detected=np.array([f.detected for _, f in rows]),
            det_score=np.array([f.det_score for _, f in rows]),
            box_w=np.array([f.box_w for _, f in rows]),
            box_h=np.array([f.box_h for _, f in rows]),
            embedding=np.stack([f.embedding for _, f in rows]),
        )
    print("listo", file=sys.stderr)


if __name__ == "__main__":
    main()
