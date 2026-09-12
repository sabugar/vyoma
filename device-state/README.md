# Device state that exists nowhere else

Everything in this folder lived only on the device's microSD card until now.
The code is in `pocket-infer-sw`; the model weights are 5.2 GB and can be
downloaded again; but the files below are small, irreplaceable, and were one
card failure away from being lost three days before the finale.

| Folder | What it is | Why it cannot be re-derived |
|---|---|---|
| `asha_knowledge_chunks/` | The 76 page files the device answers from | Extracted from the source PDF and then repaired by hand and by rule. Re-extracting gives the reversed pages and doubled letters back. |
| `asha_eval/` | The ten-question evaluation harness | The fixed set every change in this project was judged against. |
| `scripts/` | Power mode, stack restart, power measurement | Referenced by the scoped sudoers rule, which matches on exact paths. |
| `systemd/` | The service unit and the sudoers rule | How the application starts at boot and what it is allowed to do. |
| `fontsafe/` | The four real `.pcf` fonts | These become LFS pointer text on a branch switch, which is what caused the August crash loop. Restoring them from here is the fix. |

## What is still only on the card

`~/bhashini_models` — 5.2 GB of ASR, translation and speech weights. Too
large for this repository, and the one LFS object for it is missing from the
remote, so a full-image backup to external storage is the only way to make
that part safe.
