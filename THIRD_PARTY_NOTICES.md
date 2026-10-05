# Third-Party References

## FALCON

The FALCON policy contract and the optional converted robot asset are based on
`C:\Robotics\FALCON-upstream`, commit
`a967a6d8494f57777cf8d266a644ac8e45833301`. That repository includes an MIT
license. The upstream ONNX checkpoint and URDF are referenced from their local
paths by default; this repository does not copy the full upstream project.

## MoDeSuite

`C:\Robotics\MoDeSuite-reference` is a read-only local reference at commit
`470681a7ff117e487060ad52bc4f199e3fa34474`. The checkout has no explicit
`LICENSE` file. Its `asset/objects/curtain.usd` is therefore not redistributed
or copied into this repository. M1 records whether the external asset can be
loaded in Isaac Sim 5.1, but compatibility does not imply redistribution
permission.
