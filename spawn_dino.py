"""Launch BiFurca-DINO runs as independent calls on the DEPLOYED app, so they survive the local machine
sleeping or disconnecting (a `modal run` starmap is cancelled when its local parent dies; lost twice on 2026-11-06).
  modal deploy modal_train.py
  python spawn_dino.py --variants full,plain            # prints + saves call ids to logs/spawned_<profile>.json
  python spawn_dino.py --status logs/spawned_<profile>.json
Training resumes from /data/ckpt/<variant>_f<fold>_s<seed>.pt automatically; finished runs are skipped.
"""
import argparse, json, os, sys, time
import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="full,plain,no_mirror,no_coords,no_context")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    profile = os.environ.get("MODAL_PROFILE", "default")

    if a.status:
        calls = json.load(open(a.status))
        for key, cid in calls.items():
            fc = modal.FunctionCall.from_id(cid)
            try:
                r = fc.get(timeout=0)
                print(key, "DONE", r)
            except TimeoutError:
                print(key, "running")
            except Exception as e:  # noqa: BLE001 - report any remote failure
                print(key, "FAILED", type(e).__name__, str(e)[:200])
        return

    train = modal.Function.from_name("furcation-v4-train", "train")
    vol = modal.Volume.from_name("furcation-v4")
    done = set()
    for v in a.variants.split(","):
        try:
            done |= {f"{v}:{e.path.rsplit('/', 1)[-1]}" for e in vol.listdir(f"/results/{v}")}
        except Exception:  # noqa: BLE001 - directory absent = nothing finished
            pass
    calls = {}
    for v in a.variants.split(","):
        for f in a.folds.split(","):
            key = f"{v}:fold{f}_seed{a.seed}.json"
            if key in done:
                print("skip (result exists)", key)
                continue
            fc = train.spawn(variant=v, fold=int(f), seed=a.seed, epochs=a.epochs)
            calls[key] = fc.object_id
            print("spawned", key, fc.object_id, flush=True)
    os.makedirs("logs", exist_ok=True)
    out = f"logs/spawned_{profile}.json"
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(calls)
    json.dump(old, open(out, "w"), indent=1)
    print(len(calls), "calls ->", out, time.strftime("%H:%M"))


if __name__ == "__main__":
    sys.exit(main())
