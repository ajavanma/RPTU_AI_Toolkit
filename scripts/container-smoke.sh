#!/bin/sh
# Build the cpu target first. All fixtures and results stay in a temporary tree.
set -eu
image=${1:-rptu-ai-toolkit:cpu}
smoke_root=$(mktemp -d)
trap 'rm -rf "$smoke_root"' EXIT HUP INT TERM
mkdir -p "$smoke_root/raw" "$smoke_root/prepared" "$smoke_root/models" \
    "$smoke_root/results" "$smoke_root/cache"

run_rptu() {
    input_dir=$1
    output_dir=$2
    shift 2
    docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=256m \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$input_dir,dst=/input,readonly" \
        --mount "type=bind,src=$output_dir,dst=/output" \
        --mount "type=bind,src=$smoke_root/models,dst=/models,readonly" \
        --mount "type=bind,src=$smoke_root/cache,dst=/cache" \
        "$image" "$@"
}

verify_artifacts() {
    docker run --rm -i --read-only --entrypoint python \
        --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$smoke_root/models,dst=/models,readonly" \
        --mount "type=bind,src=$smoke_root/results,dst=/results,readonly" \
        "$image" - "$(id -u)" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model = Path('/models/baseline')
bundle = json.loads((model / 'bundle.json').read_text())
training = json.loads((model / 'training.json').read_text())
assert bundle['kind'] == 'rptu-segmentation-model'
assert bundle['backend'] == training['backend'] == 'torch'
assert training['device'] == 'cpu' and len(training['epochs']) == 2
assert digest(model / 'weights.safetensors') == bundle['weights']['sha256']
for directory, filename, kind in (
    ('evaluation', 'evaluation.json', 'rptu-evaluation'),
    ('predictions', 'predictions.json', 'rptu-predictions'),
):
    root = Path('/results') / directory
    report = json.loads((root / filename).read_text())
    assert report['kind'] == kind and report['device'] == 'cpu'
    assert report['model_bundle_sha256'] == digest(model / 'bundle.json')
    assert report['scenes']
    for scene in report['scenes']:
        assert digest(root / scene['path']) == scene['sha256']
hashes = {}
for root in (Path('/models'), Path('/results')):
    for path in sorted(root.rglob('*')):
        if path.is_file():
            assert path.stat().st_uid == int(sys.argv[1]), path
            hashes[str(path)] = digest(path)
print(json.dumps(hashes, sort_keys=True))
PY
}

run_rptu "$smoke_root/raw" "$smoke_root/raw" synthetic --output /output/scenes --seed 42
run_rptu "$smoke_root/raw" "$smoke_root/prepared" preprocess \
    --manifest /input/scenes/manifest.json --output /output/scenes --voxel-size 0.1
run_rptu "$smoke_root/prepared" "$smoke_root/results" inspect \
    --prepared /input/scenes/prepared.json --json
run_rptu "$smoke_root/prepared" "$smoke_root/models" train \
    --prepared /input/scenes/prepared.json --output /output/baseline \
    --device cpu --backend torch --epochs 2 --seed 42
run_rptu "$smoke_root/prepared" "$smoke_root/results" evaluate \
    --prepared /input/scenes/prepared.json --model /models/baseline \
    --output /output/evaluation --device cpu --split test
run_rptu "$smoke_root/prepared" "$smoke_root/results" predict \
    --prepared /input/scenes/prepared.json --model /models/baseline \
    --output /output/predictions --device cpu --split predict
verify_artifacts > "$smoke_root/before.json"

# Completed retries use the saved settings and leave every artifact unchanged.
run_rptu "$smoke_root/prepared" "$smoke_root/models" train \
    --prepared /input/scenes/prepared.json --output /output/baseline --resume --json
verify_artifacts > "$smoke_root/resumed.json"
cmp "$smoke_root/before.json" "$smoke_root/resumed.json"

# A repeat must reject an existing output, preserving the prior artifacts.
if run_rptu "$smoke_root/prepared" "$smoke_root/models" train \
    --prepared /input/scenes/prepared.json --output /output/baseline \
    --device cpu --backend torch --epochs 1 > "$smoke_root/repeat.log" 2>&1; then
    echo 'Expected existing output to be rejected' >&2
    exit 1
else
    test "$?" -eq 2
fi
grep -q 'output already exists' "$smoke_root/repeat.log"
verify_artifacts > "$smoke_root/after.json"
cmp "$smoke_root/before.json" "$smoke_root/after.json"

# Prepared inputs are mounted read-only, including for commands that write data.
if run_rptu "$smoke_root/prepared" "$smoke_root/results" synthetic \
    --output /input/must-not-exist --seed 42 > "$smoke_root/readonly.log" 2>&1; then
    echo 'Expected read-only input write to be rejected' >&2
    exit 1
else
    test "$?" -eq 2
fi
grep -q 'Read-only file system' "$smoke_root/readonly.log"
test ! -e "$smoke_root/prepared/must-not-exist"
echo 'CPU container workflow completed.'
