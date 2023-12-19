install:
    #!/bin/bash
    set -exo pipefail
    conda install --name torch21-cuda118 pytorch torchvision torchaudio cpuonly -c pytorch -yq
    conda run --name torch21-cuda118 --no-capture-output pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    conda run --name torch21-cuda118 --no-capture-output pip install ultralytics fiftyone supervision

run:
    #!/bin/bash
    set -exo pipefail
    python train.py   --sample_size 9000 --threshold 0.5 --name experiment --epochs 100
    python train.py   --sample_size 9000 --threshold 0.0 --name control --epochs 100

