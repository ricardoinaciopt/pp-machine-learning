# Licence and attribution

Copyright (c) Machine Learning I contributors.

## Teaching materials

Except where a different attribution or licence is stated, the original teaching text, exercises, worked explanations, course-created figures, QMD sources, notebook Markdown and generated PDF/DOCX handouts are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Course-created document templates are covered by the same licence.

You may share and adapt these materials, including commercially, provided that you give appropriate credit, link to the licence and indicate changes. Credit the named authors where supplied; otherwise use “Machine Learning I contributors”, identify the material and link to the repository or other source from which you obtained it. Retain existing attribution notices.

Full terms: [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Official licence: <https://creativecommons.org/licenses/by/4.0/>.

## Software

Original software in `src/`, `scripts/`, `tests/`, Lua rendering filters, executable notebook cells and code examples is licensed under the **MIT License**. Retain the copyright and permission notice when redistributing the software. This also covers course-authored build and environment configuration.

Full terms: [MIT License](LICENSES/MIT.txt). Official text: <https://opensource.org/license/mit>.

The distinction applies within notebooks: executable code is MIT; explanatory text and course-created figures are CC BY 4.0.

## Data and third-party material

Third-party datasets, software and attributed assets retain their own terms. The course licences do not replace those terms or claim ownership of third-party material. Keep the source attribution and transformation notices recorded in [the dataset manifest](data/manifest.csv), adjacent dataset READMEs and provenance files.

- Hotel Reservations, Concrete Compressive Strength and Wholesale Customers retain their documented CC BY 4.0 attribution. Their source-specific notices remain alongside the data.
- The repository owner confirms that `data/shared/ds1.csv`, `ds2.csv` and `ds3.csv` are toy synthetic outputs of `sklearn.datasets.make_classification` and authorises their distribution with the course under CC BY 4.0, to the extent any rights in those outputs apply. Their original generation code, parameters and random seed are unavailable; exact regeneration has not been verified. This is an owner-authorised licence for the supplied examples, not a licence inferred from scikit-learn's software licence.
- HO09's source identities and reported source licence labels remain in [its frozen manifest](data/ho09/openml_datasets.csv). The included meta-dataset contains derived experimental summaries; the course licences do not relicense the original OpenML datasets.
- Other imported teaching inputs, including Iris and the supplied historical evaluation tables, retain any applicable source terms; this notice does not assert that a new course licence replaces them.

### UCI redistribution policy

The owner has accepted redistribution of the verified Concrete Compressive Strength (UCI 165) and Wholesale Customers (UCI 292) snapshots under their source CC BY 4.0 terms. Preserve the creator/title/DOI attribution, source and licence links, transformation notices, inspection reports and frozen source/normalised hashes in `data/uci_sources.json`. Source values and row order are retained; only the documented column naming and CSV serialization changes were made. This policy applies to these two snapshots and does not change the terms of other datasets.
