# Utilities

The supported package currently provides environment diagnostics, configuration
validation, raw and prepared dataset inspection, class-ID mapping, and CPU data
preparation.

```sh
rptu doctor
rptu doctor --json
rptu config validate --config examples/config.yaml
```

Diagnostics inspect installed package metadata without initializing ML libraries.
Configuration validation checks structure and resolves paths without creating
directories. See the CLI reference and configuration pages for exact behavior.

For datasets, declared class order determines contiguous training IDs. The ignore
label stays separate, and unknown labels fail clearly. Original point identities
and row-to-voxel correspondence are preserved during preprocessing. These rules
are specified in the dataset contract.

`rptu inspect --prepared PATH` checks a portable prepared bundle on the CPU. For
Python callers, `load_prepared_manifest()` and `load_prepared_scene()` in
`rptu_ai_toolkit.prepared` expose validated metadata and numeric arrays without
ML dependencies. The reader verifies counts, mappings, and voxel label aggregation
while preserving original point order. It does not reopen raw files or prove
source provenance.

Earlier research documentation described research helpers for color maps and
Open3D point-cloud visualization. Those helpers are not a supported visualization
workflow in the installed package, and the base installation does not include
Open3D. Visualization may be added as a separate optional feature later.
