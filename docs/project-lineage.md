# Project lineage

RPTU AI Toolkit preserves both the original research code and the modern package
developed from it. Future feature development continues in
[PointCloud Workflows](https://github.com/ajavanma/PointCloud-Workflows), an
independent **private** repository. Owner-granted access is required to view or
contribute to the successor.

## Preserved branches and history

| Location | Purpose |
| --- | --- |
| [RPTU `main`](https://github.com/ajavanma/RPTU_AI_Toolkit/tree/main) | Public modern toolkit: packaging, dataset contracts, GPU/CPU models, containers, checkpoint resume, and durable training outputs. |
| [RPTU `research-legacy`](https://github.com/ajavanma/RPTU_AI_Toolkit/tree/research-legacy) | Research-era baseline before modernization, at `bcbf133df7f5158e85735a63d3b34d47e9b44074`. |
| [PointCloud Workflows](https://github.com/ajavanma/PointCloud-Workflows) | Private successor for future development, retaining the full Git history and copied branches and tags. |

The modern implementation at the transition is
[`73319745a0aacb410401df71ec890a1cecc00cf8`](https://github.com/ajavanma/RPTU_AI_Toolkit/commit/73319745a0aacb410401df71ec890a1cecc00cf8),
which includes checkpoint resume and reliable run outputs from
[PR #27](https://github.com/ajavanma/RPTU_AI_Toolkit/pull/27). RPTU `main` retains
that implementation, followed by documentation describing this transition. The
default branch remains `main`; the transition adds the legacy branch and
documentation without force-pushing or rewriting existing history.

The successor starts with a copy of the complete Git history, including all
branches and tags present at the transition. It is an independent repository,
and its subsequent development can diverge. GitHub pull-request metadata remains
in the original RPTU repository: copying Git history does not copy reviews,
discussion threads, or pull-request identities. Existing RPTU pull-request links
remain the references for work merged here.

## Historical release

The [2024 `v0.0.1` release](https://github.com/ajavanma/RPTU_AI_Toolkit/releases/tag/v0.0.1)
belongs to the original research project. Its release page and tag retain their
historical meaning; they do not identify the later modern package baseline.
Use the installation and workflow instructions on RPTU `main` for the modern
toolkit, and the legacy branch when inspecting the earlier research code.

## Documentation and future work

The [public RPTU documentation](https://rptu.gitbook.io/rptu-ai-toolkit/) continues
to describe the modern RPTU package and its existing `rptu` commands. The
[original research documentation](https://arash.gitbook.io/rptu-ai-toolkit/)
remains historical. The four README images preserve the research context.

The private successor maintains its own implementation and documentation.
Public-data recipes, larger-scene support, versioned delivery, Helm Jobs, Fleet
examples, and benchmarks are plausible successor work, rather than scheduled
RPTU releases. See the [roadmap](gitbook/todo.md) for the available RPTU workflow
and candidate follow-up stages.
