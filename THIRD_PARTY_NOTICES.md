# Third-Party References

This source release does not redistribute third-party checkpoints, robot
descriptions, USD assets, Isaac binaries, or generated video.

## FALCON

The FALCON policy contract and the optional converted robot asset are based on
the external `FALCON-upstream` checkout at commit
`a967a6d8494f57777cf8d266a644ac8e45833301`. That repository includes an MIT
license. The upstream ONNX checkpoint and URDF are acquired from that checkout
or supplied with explicit environment variables. The expected ONNX SHA256 in
the deployment contract is
`8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`.
`scripts/check_assets.py` verifies this hash when the file is present. This
repository does not copy the full upstream project or checkpoint.

## MoDeSuite

The MoDeSuite reference is a read-only checkout at commit
`470681a7ff117e487060ad52bc4f199e3fa34474`. The checkout has no explicit
`LICENSE` file. Its `asset/objects/curtain.usd` is therefore not redistributed
or copied into this repository. M1 records whether the external asset can be
loaded in Isaac Sim 5.1, but compatibility does not imply redistribution
permission.

## Isaac Sim and Isaac Lab

Isaac Sim, Isaac Lab, their extensions, and the NVIDIA runtime are installed by
the user under the applicable NVIDIA terms. No binary, driver, CUDA package,
or simulator checkout is part of this repository. Replay scripts accept
`ISAACLAB_ROOT`/`ISAACLAB_LAUNCHER`, so an installation path is never embedded
in new source code.

## Acquisition and reproducibility

Use the external repositories' official distribution channels and record the
checkout commit, local file path, and SHA256 in a local audit manifest. Do not
commit credentials, private keys, checkpoints, URDF/USD files, crash dumps, or
rendered videos. Generated rubber-hand USD is a local conversion product and
is ignored by Git.
