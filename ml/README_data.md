\# Topic Classifier — MLOps Final Project



FastAPI service that classifies a short text as `hockey` or `space`, using a

TF-IDF + LogisticRegression model trained on a subset of 20 newsgroups.



\## Repo layout



\- `app/` — FastAPI service

\- `ml/` — data prep, training, configs, registry, docs

\- `tests/` — pytest suite

\- `.github/workflows/` — CI + staging/prod image builds

\- `data/raw/` — DVC-tracked datasets (v1, v2)



\## Pipeline: commit → build → test → staging → prod



1\. Feature branch: `feature/<name>` → PR into `dev`.

2\. PR to `dev`/`staging`/`main` triggers `ci.yml`:

&#x20;  - ruff, black --check

&#x20;  - pytest

&#x20;  - docker build

&#x20;  CI must pass before merge.

3\. Merge into `staging` → `deploy-staging.yml` builds staging image using

&#x20;  staging GitHub secrets.

4\. Merge into `main` → `deploy-prod.yml` builds production image using

&#x20;  production GitHub secrets.



Branch flow: `feature → dev → staging → main`.



\## Quickstart



```bash

python -m venv .venv \&\& source .venv/bin/activate

pip install -r requirements-dev.txt

python -m ml.prepare\_data

dvc add data/raw/v1 data/raw/v2

python -m ml.train --config baseline.yaml --data-version v1

python -m ml.train --config improved.yaml --data-version v2

uvicorn app.main:app --reload

