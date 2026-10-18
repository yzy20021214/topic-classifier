# Data

## Source

`sklearn.datasets.fetch_20newsgroups`, categories:
`rec.sport.hockey` and `sci.space`.

## Versions

- **v1** — raw text, headers/footers/quotes kept. Commit: `<fill in>`
- **v2** — cleaned text, `remove=("headers", "footers", "quotes")`. Commit: `<fill in>`

## What changed v1 → v2

- Removed email headers, footers and quoted replies.
- Result: less leakage from metadata, more realistic text classification.

## Reproduce

```bash
python -m ml.prepare_data
dvc add data/raw/v1 data/raw/v2
dvc push```