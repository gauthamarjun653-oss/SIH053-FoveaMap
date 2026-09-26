# SalsaNext model provenance

Source: official SalsaNext repository, https://github.com/TiagoCortinhal/SalsaNext
Pretrained archive linked by that repository:
https://drive.google.com/file/d/1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD/view

Downloaded archive: `salsanext-upstream.zip`.
SHA256: cfbbb5403f9438ccc1ff0fac6b64c3f5f21a1881e76e4172a45a0e077624e0c5

Its `pretrained/SalsaNext` training checkpoint was extracted as
`salsanext-upstream-checkpoint.pt`. Architecture and dataset settings are saved
as `salsanext-arch.yaml` and `salsanext-data.yaml`. The channel normalization and
64x2048 projection agree with the implementation.

The legacy checkpoint was read with PyTorch `weights_only=True` using explicitly
allowlisted legacy NumPy and training metadata types; no unrestricted pickle
load was used. Only its tensor state dictionary was saved to
`salsanext_state_dict.pt`, stripping the DataParallel `module.` prefix. Every
parameter name and shape passed `SalsaNext(20).load_state_dict(strict=True)`.
The runtime loads only this state dictionary with `weights_only=True` and rejects
missing/unexpected parameters. It never substitutes random weights silently.

The current local environment uses torch 2.12.1+cu130 on an RTX 3050 6 GB GPU.
Seq 09/10 are training-split diagnostics, not held-out validation data.
