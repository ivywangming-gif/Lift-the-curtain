# Checkpoint replacement failure: read-only diagnostic

Snapshot: 2026-11-06T18:19:40.297278+00:00 to 2026-11-06T18:19:40.355391+00:00 (UTC).

Run: `learning_campaign_3e48cd09d5df3559fdb6`. The trainer was still progressing; these counts are a point-in-time snapshot, not the final run status. Queue counts: `{"BLOCKED_WITH_REASON": 1260, "EVALUATED": 99, "FAILED_WITH_REASON": 19, "PLANNED": 141, "TRAINING": 1}`.

The three requested GRU failures occurred after successful checkpoint serialization, during `partial.tmp.pt` → `partial.pt` replacement. All six files deserialize on CPU, all model/optimizer/RNG tensors are finite, and size/mtime remained stable throughout each read. The temporary checkpoint contains exactly one additional completed epoch, with the committed curve as its exact prefix. The fixed total is 80 epochs.

| GRU seed | Committed next epoch | Temporary next epoch | Committed / temporary bytes |
| --- | ---: | ---: | ---: |
| 40702 | 73 | 74 | 745,426 / 745,490 |
| 40703 | 47 | 48 | 743,442 / 743,506 |
| 40704 | 40 | 41 | 742,930 / 742,994 |

All six files have `Archive` attribute (32), with no read-only flag. The three directory ACLs show authenticated-user Modify and administrator/System FullControl, with no explicit deny observed. These checks do not prove the effective runtime token or exclude a transient sharing conflict.

`train_benchmark.py:392–396` calls `torch.save` and then `Path.replace`; there is no separate `save_checkpoint` function. Local PyTorch 2.7.0+cu128 exits its ZIP-writer context before the replacement. The fit code has no explicit retained Python file handle. These directories had already committed many epochs successfully, so the evidence does not support a simple permanent read-only or corrupt-checkpoint explanation.

A transient Windows sharing/delete denial is consistent with the symptoms, but its cause is unconfirmed. A scanner, runtime identity interaction, or native/external handle timing cannot be distinguished by this read-only snapshot. No lock owner was observed. Sandbox process enumeration did not expose the host trainer and must not be interpreted as evidence that it had exited.

All 10 current frozen source hashes still match. Protocol SHA: `f9d1757d028c114eb4e149c71adcc57094cb397ea0b5ad723698fa04fe11328c`. No source/protocol/run/queue/failed file was changed, and no rename test or training was attempted.

## Safe same-protocol recovery

1. Wait for the current owned trainer to exit and its resource lock to be released; do not modify active fits.
2. Archive exact bytes and SHA of the first-pass queue, status, frozen protocol, failed directories including partial.pt, partial.tmp.pt, failure.json, and source manifest before retries.
3. Retain the same frozen protocol, code hashes, data, seeds, device and fixed 80 epochs; use a fresh valid exclusive training window.
4. Run the existing run command against the same output. It skips EVALUATED/BLOCKED_WITH_REASON and retries FAILED_WITH_REASON without manual queue edits.
5. Resume only the committed partial.pt: restore model, optimizer, CPU/CUDA RNG and curve; do not promote partial.tmp.pt or select an epoch by performance.
6. For these GRU seeds the committed next_epoch values are 73,47,40; recompute the one uncommitted epoch and finish the frozen total. CUDA bitwise identity is not promised by the existing nondeterministic implementation.
7. Preserve any second failure independently. If I/O code changes become necessary, audit/version them separately; never silently rewrite the original frozen protocol code hash.

## Per-file hashes

| Seed | File | SHA-256 |
| --- | --- | --- |
| 40702 | `partial.pt` | `a15caefe7e86085bb1dc5d4e8c661e745bb6b0128b70435a670d76d76451c367` |
| 40702 | `partial.tmp.pt` | `b7e5c427934e0565db63c9980428357cf22839bd0b20de6364ad8ba1de15fd26` |
| 40702 | `failure.json` | `8e22b8310365de02d71e59c39258fd86c735219807e70815dd6231bb2843e330` |
| 40703 | `partial.pt` | `3c652aa0a013023a5dd6c104aeb6e8d289e3ccd08f7a6c0cfeb11ed1d3232637` |
| 40703 | `partial.tmp.pt` | `fe5645a8edfd26ad6a16c18186458281145e74bd24a1c21c4c718f9477d64917` |
| 40703 | `failure.json` | `590ea572457190e4c8407637b71104c313154b81b0f96ca568173eb4040d8c02` |
| 40704 | `partial.pt` | `8125e74728c16ea43c74b0d61a56571eeb2010e7ecc250feb87177aee7c5c756` |
| 40704 | `partial.tmp.pt` | `76fd95ca4575ddc0d3653f43d1ef5d1bdbff5bf2198b24ddb36429b18092b274` |
| 40704 | `failure.json` | `3c9edbd081e91bc3b1c6fd8899a90e4010b79a928c15e41409e93026505e3ba6` |

Full attributes, timestamps, stable-read checks, source hashes and the contemporaneous queue hash are in [the JSON evidence](CHECKPOINT_RENAME_DIAGNOSTIC_20261106.json).
