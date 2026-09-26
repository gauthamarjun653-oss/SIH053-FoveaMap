# Stage 0 — Provision and data staging

**Status: DONE** (2026-08-27)

Purpose: make S3 the complete canonical data store and the Azure VM the CPU
working copy, so every downstream stage is unblocked.

## Data inventory (verified 2026-08-27)

S3 canonical store, `s3://sih-perception-dataset-882040517559/`:

| Prefix | Contents | Status |
|---|---|---|
| `sequences/00/` | velodyne 4,541 + labels 4,541 + calib.txt | complete |
| `sequences/02/` | velodyne 4,661 + labels 4,661 + calib.txt | complete |
| `sequences/05/` | velodyne 2,761 + labels 2,761 + calib.txt | complete |
| `sequences/08/` | velodyne 4,071 + labels 4,071 + calib.txt | complete |
| `poses/` | 11 pose files (archive top-level layout) | complete |
| `weights/` | `salsanext_state_dict.pt` (27 MB, 312 tensors) + reference `arch_cfg.yaml` / `data_cfg.yaml` | complete |

Azure VM working copy, `~/kitti/` (dev sequence 00 + validation sequence 08
only — 02/05 are training sequences and are not needed on any machine, since
the project uses pretrained weights and never trains):

| Path | Contents | Status |
|---|---|---|
| `sequences/00/` | velodyne 4,541, labels 4,541, calib.txt, poses.txt | complete |
| `sequences/08/` | velodyne 4,071, labels 4,071, calib.txt, poses.txt | complete |

Repo working copy: `weights/salsanext_state_dict.pt` present (gitignored).

Nothing is missing. Every artifact the execution plan needs exists in S3, and
everything the CPU stages need also exists on this VM.

## Account migration (2026-08-27, later the same day)

The canonical store moved to the new AWS account **882040517559** (bucket
`sih-perception-dataset-882040517559`, us-east-1). The data was relayed
through the Azure VM (old profile read, new profile write) with zero changes
to the old account, and every per-prefix object count verified equal. The VM's
`[default]` AWS profile now points at the new account; the old account remains
available as `[oldaccount]`. Final totals: old 32,087 objects / 39.66 GB; new
32,090 / 40.26 GB (the extra three are the synced `runs/` artifacts - the
byte delta matches them exactly).

**Old account retired.** Deletion of the old bucket was owner-approved
2026-08-27; it could not be executed from the VM (the scoped user lacked
version-delete permissions on the versioned bucket) and was handed to the
owner's console. The old-account access key on the VM has since become
invalid (`InvalidAccessKeyId`), consistent with the owner tearing down the
old account's IAM/data - final deletion cannot be re-verified from here. The
`[oldaccount]` profile in `~/.aws/credentials` is now dead weight and safe to
remove.

## What was done

- Azure VM `foveamap-staging-vm` provisioned and bootstrapped
  (`infra/azure/bootstrap_vm.sh`); venv with CPU-only torch.
- SemanticKITTI labels fetched from semantic-kitti.org (179 MB archive),
  placed locally for 00/08 (counts match scans exactly), uploaded to S3 for
  00/02/05/08 (counts verified object-by-object).
- SalsaNext pretrained weights fetched (owner approved 2026-08-27), converted
  once offline from the training checkpoint to a plain state dict, verified
  with `torch.load(weights_only=True)` (T-28) plus a Segmenter init and smoke
  inference, staged locally and in S3.

## Notes

- The archive ships poses at top-level `poses/XX.txt`; every synced copy must
  place them at `sequences/XX/poses.txt` (spec 11.2). Done for 00 and 08.
- `data_root` is `~/kitti` in `configs/pipeline.yaml`; the loader expands `~`.
  Never hardcode absolute paths.
