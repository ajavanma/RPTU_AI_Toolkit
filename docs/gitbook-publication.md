# GitBook documentation publication

The current documentation is published at
[RPTU AI Toolkit](https://rptu.gitbook.io/rptu-ai-toolkit/).
The site was first published on 2026-10-13 on GitBook's free **Basic** plan; no trial
was activated. Initial change requests 1 and 2 covered the CPU packaging and raw
dataset workflow at `4601febadbbe67777da75a57949a22f5e04bed4b`.

Public verification loaded all twelve pages and checked their titles, navigation,
rendered commands, and capability boundaries. Fresh page loads confirmed the final
research-history wording on the homepage, Train, Test, and Model pages. Content
links point to public destinations; GitBook's signed-in editor toolbar is separate
from the documentation content.

| Item | Value |
| --- | --- |
| Organization ID | `vniS8Hj53VXYZdAU1Epj` |
| Site ID | `site_pIX7d` |
| Space ID | `ajFUWYe5yIaghJc56ivh` |
| Public URL | <https://rptu.gitbook.io/rptu-ai-toolkit/> |
| Publication mode | Manual Markdown snapshot; no Git Sync |
| Documentation source baseline | GPU/CPU workflow with checkpoint resume and durable training runs; see this file's Git history for the source revision |

The [source pages](gitbook/README.md) describe the GPU/CPU toolkit, prepared-data
workflow, model bundles, resumable training, evaluation, and point-label prediction. Update
existing pages in a reviewed GitBook change request and keep the stable page slugs.
This public site continues to document RPTU's modern `main` baseline following
the [project transition](project-lineage.md). The private PointCloud Workflows
successor maintains its own documentation; successor implementation and private
documentation are not published on this site. Keep the RPTU name, commands,
existing URLs, and research-history references here.
Record the completed source merge and public verification in the modernization
plan. A prior publication issue left public pages stale after reader change
request 3 was merged, despite the preview showing the new content; compare the
public output when publishing this update.

The [original research site](https://arash.gitbook.io/rptu-ai-toolkit/) was not
accessible for editing through the available account and remains unchanged. The
new site was created with the user's authorization in the available organization;
it does not replace or redirect the original address.

## Source files and navigation

`docs/gitbook/` contains twelve content pages and a navigation reference. The
published page order, titles, and slugs are:

| File | Page title | Public path after the site base |
| --- | --- | --- |
| `README.md` | RPTU AI Toolkit | Homepage `/` |
| `quickstart.md` | GPU and CPU quickstart | `quickstart` |
| `cli-reference.md` | CLI reference | `cli-reference` |
| `dataset-contract.md` | Dataset contract | `dataset-contract` |
| `configuration.md` | Configuration | `configuration` |
| `development.md` | Development and tests | `development` |
| `preprocess.md` | Preprocess | `preprocess` |
| `train.md` | Train | `train` |
| `test.md` | Test and evaluation | `test` |
| `utils.md` | Utilities | `utils` |
| `model.md` | Model | `model` |
| `todo.md` | Roadmap | `todo` |

[SUMMARY.md](gitbook/SUMMARY.md) records this order for maintainers. The native ZIP
importer did not recognize it as navigation: it imported a separate Summary page.
That extra page was removed through GitBook's recoverable page deletion. The
importer also used filenames as initial page titles, so the twelve titles and
navigation order were corrected manually and the overview was selected as the
homepage.

## Manual publication workflow

1. Select the merged source revision that the documentation will describe. Edit
   the twelve Markdown pages under `docs/gitbook/`, keeping unavailable features
   clearly marked as planned. Update `SUMMARY.md` as the navigation reference.
2. For a full import into a new space, create a ZIP with the twelve content files
   at its root and use GitBook's native Markdown/ZIP import. Exclude `SUMMARY.md`
   from the upload; it is not applied as navigation by this import flow. Do not
   include generated data, validation logs, or local helper scripts.
3. Rename the imported filename-based titles to the table above. Check their URL
   slugs, reorder the pages manually, and select the RPTU AI Toolkit overview as
   the homepage. For updates to existing pages, edit those pages rather than
   importing duplicates.
4. Review formatting, code blocks, lists, tables, and page references in a GitBook
   change request. If pasting formatted HTML for a correction, use links under
   `https://rptu.gitbook.io/rptu-ai-toolkit/`, not private editor/change-request URLs.
5. Merge the reviewed change request and publish using the site's Basic plan.
   Confirm the public homepage, navigation, affected page URLs, and rendered
   content after publication. Keep the original research site link labeled
   historical.
6. Record the published source revision and any manual corrections here. Changes
   to repository files do not automatically update the public site: no Git Sync
   integration is configured.

The initial import and follow-up corrections retained the merged CPU capability
boundary. The Train, Test, Utils, and Model pages now refer to earlier research
documentation without implying they occupy the original site's URLs. The
Utilities page uses plain CLI, configuration, and dataset-contract references to
match the manually corrected published content.
