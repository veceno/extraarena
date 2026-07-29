# ExtraLR Nemesis Lite Preview

Status: `preview` (not approved for human matchmaking)

## Training corpus

- 19,848 complete Arena simulations, balanced at 9,924 rows per checkpoint:
  `u29250` (post-B) and `h299` (post-C policy without Ultra assists).
- 1,464 unordered exact deck-pair groups.
- Every retained seed quartet contains both model seats and both starting
  sides.
- 19,848 unique namespaced battle IDs; zero truncated/error/invalid-action
  rows.
- 50-card catalog SHA-256:
  `2d4e28c0f774097538c0c90ed341152155a1bb0622aa598c4677c3de51b39315`.
- Legacy requested levels above 2 for simplified cards 11/12/13 were
  normalized to the effective level 2 actually used by the engine. The source
  rows, transformation counts and hashes are retained in
  `TrainV3.5/runs/nemesis_lite_preview_u29250_h299_v1/dataset_manifest.json`.

The final trainer split is grouped by unordered exact decks with levels:
13,804 train / 2,884 validation / 3,160 test rows, with 1,024 / 220 / 220
groups and no group overlap.

## Preview result

On the grouped test set:

| Model | Accuracy | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| Nemesis Lite Preview | 83.96% | 0.3690 | 0.2285 | 0.0127 |
| Starter-only baseline | 55.35% | 0.6960 | 0.4946 | 0.0031 |
| Train class-prior baseline | 50.28% | 0.6960 | 0.5003 | 0.0023 |

Checkpoint slices are descriptive rather than independent model holdouts:

- `u29250`: 87.29% accuracy over 1,660 test rows;
- `h299`: 80.27% accuracy over 1,500 test rows.

The architecture is permutation-invariant within each deck and
swap-equivariant by construction. Test swap probability drift is at most
`1.19e-7`; PyTorch-to-ONNX probability parity drift is at most `2.68e-7`.

## Hard limitation

Class counts are 9,866 P1 wins / 2 draws / 9,980 P2 wins. The grouped test set
contains one draw; Preview assigns it draw probability `8.0e-7` and predicts
zero draws overall. Consequently:

- the reported 83.96% is effectively a non-draw winner metric;
- the `draw` output exists for contract compatibility but is not calibrated;
- promotion requires a targeted draw/stalemate lane or an explicit binary
  winner/no-winner policy.

The model estimates outcomes under the sampled non-Ultra V5 policy mixture.
It must not be interpreted as a calibrated human-vs-human probability until
fine-tuned and evaluated on human-vs-human, player-disjoint and chronological
holdouts.

## Artifacts

- Dataset:
  `TrainV3.5/runs/nemesis_lite_preview_u29250_h299_v1/nemesis_lite_preview.jsonl`
  (`baff2161168aeabdee3e2137da7260ff840bead032f54ff6a58c8e8abd8f016f`)
- Training checkpoint:
  `TrainV3.5/runs/nemesis_lite_preview_u29250_h299_v1/trained/extra_lr_nemesis_lite_preview.npz`
- ONNX:
  `ai/models/extra_lr_nemesis_lite_preview.onnx`
  (`e71e29f6fddae54e471a495474a8b99ff266b61bd8688cc2576b93b9841a78df`)
- Runtime: `ai/nemesis_lite_preview.py`
