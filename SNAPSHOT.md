# pocket-infer-sw — working-state snapshot

This branch is a **snapshot of the running device**, not the upstream history.

The upstream repository stores its platform assets in Git LFS —
`bhashini_models.zip` (1.6 GB), `nmt_trans.zip` (711 MB),
`flite_voices.zip` (264 MB) and others, about 2.7 GB in total. GitHub
refuses a push whose LFS pointers have no objects behind them, and the free
LFS allowance is 1 GB, so the real history cannot be mirrored here. Those
three asset directories are therefore excluded:

    rootfs/roles/app/files/
    rootfs/roles/indic/files/
    rootfs/roles/initial/files/

Everything else is the live working tree, byte for byte.

`.gitattributes` has been removed on purpose. It marked `*.pcf` as LFS, and
the four Devanagari/ForkAwesome bitmap fonts are only ~600 KB in total, so
they are committed here as ordinary files. That also removes a real hazard:
an LFS pointer checked out in place of a real font makes the UI process
crash-loop with `ValueError: Unknown magic number b'vers'`.

`device-scripts/` holds the host helper scripts that live in `/home/ubuntu`
and are not part of either application repository.
