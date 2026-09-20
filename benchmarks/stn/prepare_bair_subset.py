#!/usr/bin/env python
"""
Prepare a fixed BAIR prefix subset (and the VGG/LPIPS loss weights) inside a personal directory.

  data     Stream the first bytes of the Hugging Face archive `taldatech/bair_256/bair_256_ours.tar.gz`
           (a single 126.6 GB gzip stream) and keep the first N complete `train/<episode>/` directories in
           ARRIVAL ORDER: the first TRAIN_N become `train/`, the next VAL_N become `val/`. This is a fixed
           PREFIX subset with an INTERNAL validation split. It is neither a random sample nor the official
           validation set. The connection is closed as soon as enough complete episodes were seen, and
           the run stops (without pulling more) if a byte cap is reached or the stream fails.
  unpad    Rename zero-padded episode folders (05498 -> 5498) so datasets/bair_ds.py:BAIRImage can find them.
  weights  Download the torchvision VGG16 weights and the LPIPS linear head used by utils/loss_functions.py
           (LossLPIPS, i.e. recon_loss_type == "vgg"), verifying size and hash.

A complete episode is 30 PNG frames (00000.png..00029.png) plus actions/dones/metadata/rewards .pkl.
A manifest with the source revision, episode list, per-file sha256 and an aggregate hash is written.

    python prepare_bair_subset.py data --out DIR [--train-episodes 400 --val-episodes 40 --max-bytes 2500000000]
    python prepare_bair_subset.py weights --torch-home DIR --lpips-dir DIR
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time

import requests

REPO_ID = "taldatech/bair_256"
ARCHIVE = "bair_256_ours.tar.gz"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ARCHIVE}"
API = f"https://huggingface.co/api/datasets/{REPO_ID}"
FRAMES = 30
PKLS = ("actions.pkl", "dones.pkl", "metadata.pkl", "rewards.pkl")
WANT = {f"{i:05d}.png" for i in range(FRAMES)} | set(PKLS)
MEMBER = re.compile(r"^bair_256_ours/(\w+)/(\d+)/([^/]+)$")

VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG_BYTES, VGG_SHA256_PREFIX = 553433881, "397923af"
LPIPS_URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
LPIPS_BYTES, LPIPS_MD5 = 7289, "d507d7349b931f0638a25a48a722f98a"


class CountingReader(io.RawIOBase):
    """Counts the compressed bytes actually read from the network and enforces a hard cap."""

    def __init__(self, raw, cap):
        self.raw, self.cap, self.count = raw, cap, 0

    def readable(self):
        return True

    def readinto(self, b):
        if self.count >= self.cap:
            raise RuntimeError(f"byte cap of {self.cap} reached")
        data = self.raw.read(len(b))
        b[:len(data)] = data
        self.count += len(data)
        return len(data)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def cmd_data(args):
    out = os.path.abspath(args.out)
    total = args.train_episodes + args.val_episodes
    os.makedirs(out, exist_ok=True)
    meta = requests.get(API, timeout=60).json()
    head = requests.head(URL, allow_redirects=True, timeout=60)
    manifest = {"kind": "fixed PREFIX subset with INTERNAL validation split (not random, not the official validation set)",
                "source": {"hf_repo": REPO_ID, "url": URL, "revision_sha": meta.get("sha"), "last_modified": meta.get("lastModified"),
                           "license": (meta.get("cardData") or {}).get("license"), "archive_bytes": int(head.headers.get("content-length", 0)),
                           "etag": head.headers.get("etag") or head.headers.get("x-linked-etag")},
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha256_bytes(open(__file__, "rb").read()),
                "rule": f"first {args.train_episodes} complete train episodes in archive order -> train/, next {args.val_episodes} -> val/",
                "max_bytes": args.max_bytes, "episodes": [], "skipped_incomplete": []}
    status = "unknown"
    partial, accepted, other_splits, n_members = {}, [], set(), 0
    resp = requests.get(URL, stream=True, timeout=(30, 120))
    resp.raise_for_status()
    reader = CountingReader(resp.raw, args.max_bytes)
    t0 = time.time()
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader, buffer_size=1 << 20), mode="r|gz") as tf:
            for m in tf:
                n_members += 1
                mm = MEMBER.match(m.name) if m.isfile() else None
                if not mm:
                    continue
                split, ep, name = mm.groups()
                if split != "train":
                    other_splits.add(split)
                    continue
                if name not in WANT:
                    continue
                partial.setdefault(ep, {})[name] = tf.extractfile(m).read()
                if len(partial[ep]) == len(WANT):
                    files = partial.pop(ep)
                    accepted.append((ep, files))
                    if len(accepted) == total:
                        manifest["prefix_end_member_index"] = n_members
                        status = "complete"
                        break
    except Exception as exc:  # noqa: BLE001 - stop and report; never continue pulling
        status = f"stopped: {type(exc).__name__}: {exc}"
    finally:
        resp.close()
    manifest["stream"] = {"status": status, "compressed_bytes_read": reader.count, "seconds": round(time.time() - t0, 1),
                          "members_seen": n_members, "accepted_complete_episodes": len(accepted),
                          "partial_episodes_left": {ep: len(f) for ep, f in partial.items()}, "other_splits_seen": sorted(other_splits)}
    if status != "complete":
        json.dump(manifest, open(os.path.join(out, "manifest_INCOMPLETE.json"), "w"), indent=2)
        print(json.dumps(manifest["stream"], indent=1))
        print("STOPPED: the requested subset could not be cut as expected; nothing more was downloaded.")
        return 1

    agg = hashlib.sha256()
    for i, (ep, files) in enumerate(accepted):
        split = "train" if i < args.train_episodes else "val"
        d = os.path.join(out, split, ep)
        os.makedirs(d, exist_ok=True)
        entry = {"episode": ep, "split": split, "order": i, "files": {}}
        for name in sorted(files):
            with open(os.path.join(d, name), "wb") as f:
                f.write(files[name])
            h = sha256_bytes(files[name])
            entry["files"][name] = {"bytes": len(files[name]), "sha256": h}
            agg.update(f"{split}/{ep}/{name}:{h}\n".encode())
        manifest["episodes"].append(entry)
    manifest["aggregate_sha256"] = agg.hexdigest()

    from PIL import Image
    probe = Image.open(os.path.join(out, "train", accepted[0][0], "00000.png"))
    manifest["verify"] = {"frame_size": list(probe.size), "frame_mode": probe.mode,
                          "train_episodes": sum(e["split"] == "train" for e in manifest["episodes"]),
                          "val_episodes": sum(e["split"] == "val" for e in manifest["episodes"]),
                          "train_images": args.train_episodes * FRAMES, "val_images": args.val_episodes * FRAMES,
                          "disjoint": len({e["episode"] for e in manifest["episodes"]}) == total,
                          "total_bytes_on_disk": sum(f["bytes"] for e in manifest["episodes"] for f in e["files"].values())}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps({"stream": manifest["stream"], "verify": manifest["verify"], "aggregate_sha256": manifest["aggregate_sha256"],
                      "source_revision": manifest["source"]["revision_sha"]}, indent=1))
    return 0


def cmd_unpad(args):
    """BAIRImage does str(int(folder)), so archive folders like `05498` are not found. Rename them to `5498`
    (the episode number is unchanged) and record archive name -> disk name in the manifest. Idempotent."""
    out = os.path.abspath(args.out)
    path = os.path.join(out, "manifest.json")
    manifest = json.load(open(path))
    renamed = 0
    for e in manifest["episodes"]:
        disk = str(int(e["episode"]))
        e["dir_on_disk"] = disk
        src, dst = os.path.join(out, e["split"], e["episode"]), os.path.join(out, e["split"], disk)
        if disk != e["episode"] and os.path.isdir(src):
            assert not os.path.exists(dst), dst
            os.rename(src, dst)
            renamed += 1
    manifest["dir_names"] = ("leading zeros removed from episode folder names because datasets/bair_ds.py:BAIRImage uses "
                             "str(int(name)); files inside are unchanged; the aggregate hash is over archive names")
    json.dump(manifest, open(path, "w"), indent=2)
    counts = {sp: len(os.listdir(os.path.join(out, sp))) for sp in ("train", "val")}
    print(json.dumps({"renamed": renamed, "episode_dirs": counts}))
    return 0


def fetch(url, path, want_bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        got = 0
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                got += len(chunk)
                if got > want_bytes:
                    raise RuntimeError(f"{url}: more than the expected {want_bytes} bytes; aborted")
                f.write(chunk)
    if got != want_bytes:
        raise RuntimeError(f"{url}: got {got} bytes, expected {want_bytes}")
    os.replace(path + ".part", path)


def cmd_weights(args):
    vgg = os.path.join(args.torch_home, "hub", "checkpoints", "vgg16-397923af.pth")
    lpips = os.path.join(args.lpips_dir, "vgg.pth")
    res = {}
    if not os.path.exists(vgg):
        fetch(VGG_URL, vgg, VGG_BYTES)
    h = hashlib.sha256(open(vgg, "rb").read()).hexdigest()
    assert h.startswith(VGG_SHA256_PREFIX), h
    res["vgg16"] = {"path": vgg, "bytes": os.path.getsize(vgg), "sha256": h}
    if not os.path.exists(lpips):
        fetch(LPIPS_URL, lpips, LPIPS_BYTES)
    md5 = hashlib.md5(open(lpips, "rb").read()).hexdigest()
    assert md5 == LPIPS_MD5, md5
    res["lpips_head"] = {"path": lpips, "bytes": os.path.getsize(lpips), "md5": md5}
    json.dump(res, open(os.path.join(args.torch_home, "weights_manifest.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    p.add_argument("--out", required=True)
    p.add_argument("--train-episodes", type=int, default=400)
    p.add_argument("--val-episodes", type=int, default=40)
    p.add_argument("--max-bytes", type=int, default=2_500_000_000, help="hard cap on compressed bytes read")
    p = sub.add_parser("unpad")
    p.add_argument("--out", required=True)
    p = sub.add_parser("weights")
    p.add_argument("--torch-home", required=True)
    p.add_argument("--lpips-dir", required=True)
    args = ap.parse_args(argv)
    return {"data": cmd_data, "unpad": cmd_unpad, "weights": cmd_weights}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
