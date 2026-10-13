# Train

Training is planned and is not implemented in the current installed CLI.
`rptu train` returns an explicit not-implemented error with exit status `2`.

The supported workflow currently prepares and validates data on the CPU. Start
with the [quickstart](quickstart.md), and use the [dataset contract](dataset-contract.md)
to supply labeled train and validation scenes.

A fresh WarpConvNet model, training loop, checkpoint resume, and validation will
be developed after establishing a supported GPU environment. No original trained
weights or research datasets are included, so the new pipeline will establish its
own baseline.

Earlier research documentation described PyTorch Lightning and MinkowskiEngine
research scripts. Those scripts are incomplete and do not provide a supported
training workflow. Their old function signatures are not the new CLI contract.

See [Model](model.md) and the [roadmap](todo.md) for the intended direction.
