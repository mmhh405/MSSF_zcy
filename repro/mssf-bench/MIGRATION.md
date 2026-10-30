# Project relocation (2026-10-30)

The project, Git history, working changes, checkpoints, and run logs now live in:

`/home/d514/桌面/zcy/MSSF_zcy`

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
```

The previous path `/home/d514/projects/MSSF_zcy` is a compatibility symlink to this directory. There is one physical project checkout.

The Conda environment stays at `/home/d514/miniconda3/envs/mssf-bench`; its editable mmdet3d package now points directly to the new source directory. VoD remains on NAS, linked through `data/vod`.

All 1322 existing regular files were verified by SHA-256 before and after the move, including Git metadata. Modes, ownership and symlink targets were preserved. The environment README was then updated to the new path.

A 16-frame validation run from the new Chinese path completed successfully with two data-loader workers and exit code 0. Predictions and EAA/DC metrics were produced. Full-set validation and training were not started. See `migration-verification.json` for the paths and verification details.

Historical validation records retain their original paths; the compatibility symlink keeps them accessible. Git branch, commit, remotes and working changes were preserved. No commit or push was performed during migration.
