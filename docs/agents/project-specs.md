# Project specs, tools, build & conventions

> **Zweck.** Jeder projektspezifische Fakt für Agenten und Beitragende.
> Geladen vom Metaprompt in [`AGENTS.md`](../../AGENTS.md), zusammen mit
> [`research.md`](research.md) und [`devmind.md`](devmind.md).
>
> **Evidenzregeln.** Jede Aussage ist belegt (`file:line`, Git-Historie,
> CI-Config oder ausgeführtes Kommando). Kommandos tragen
> `<!-- verified YYYY-MM-DD: <Beobachtung> -->` oder
> `<!-- UNVERIFIED — source: <Fundstelle> -->`.

## Build Commands

Reines Python-Paket (setuptools), kein Compile-Schritt.

```bash
# Installation (Kern: nur numpy)
pip install -e .               # oder: pip install -e ".[test]"

# Modellfreie Mathematik-Tests (8 Tests, ohne Download)
python -m pytest tests -q

# Backends optional
pip install -e ".[hf]"         # transformers
pip install -e ".[nanogpt]"    # tiktoken + nanoGPT-Checkpoint

# CLI
provenance-audit --help
```
<!-- verified 2026-11-19: pip install aus Git in frischem Venv exit 0; pytest 8 passed; provenance-audit --help exit 0 -->
<!-- UNVERIFIED — source: pyproject.toml [project.optional-dependencies], Extras nicht in dieser Session installiert -->

`pyproject.toml` ist die einzige Build-Definition
(`[build-system] setuptools>=68`, `[project.scripts] provenance-audit`).
Es gibt kein `requirements.txt` und keine Lockdatei.

## Specifications & references

- **`docs/METHOD.md`** — was gemessen wird, Zell-Konstruktion, Protokoll.
- **`docs/METRICS.md`** — Definition von Halluzination im Sinne des Pakets:
  nicht durch Training gedeckt **und** nicht durch den Prompt gegeben.
- **`docs/PORTS.md`** — read-only Ports der drei Achsen in fremde
  Codebasen; keine Gewichte werden verändert.
- **`provenance_audit/axes.py:1-8`** — die drei Achsen mit ihren Kontrasten:
  `context` (C vs I), `familiarity` (F+ vs F-), `own text` (Modell vs Nutzer).
- **`results/summary.json`** — eingefrorene Zahlen, die die README zitiert;
  aus Entwicklungs-Geräten (CPU, float32), kein Benchmark-Score
  (`results/README.md`).

## Linting / Formatting

Nichts erzwungen: keine `ruff`/`black`/`flake8`-Config, kein CI
(verifiziert 2026-11-19: kein `.github/`). Der Stil wird per Konvention
gehalten (siehe unten).

## Coding Conventions

- **Englische Docstrings** in allen Modulen, mit einer präzisen
  Begriffsdefinition im Kopf (`core.py:1-8`, `indicator.py:1-9`).
- **`from __future__ import annotations`** in jedem Modul
  (`core.py`, `axes.py`, `indicator.py`, `cli.py`).
- **Kern ist modellfrei**: `core.py` nimmt numpy-Arrays, ein „Backend" muss
  nur Hidden States liefern (`core.py:1-6`) — deshalb laufen die Tests ohne
  Modell.
- **Achsen sind lineal + Leave-one-out** auf Minimalpaaren; nichts wird
  trainiert (`axes.py:1-10`).
- **Nur kalibrierte Achsen dürfen freisprechen** — das ist als Commit-Norm
  festgehalten (`indicator: only calibrated axes may justify an answer`).

## Commit Message Style

`Bereich: Gegenstand`, englisch:

```
cleanup: correct type hints and unused imports
indicator: only calibrated axes may justify an answer
adapters: tunable wording, extractor hook; fix 1-D axis scoring
provenance-audit: three provenance axes and a one-pass hallucination indicator
```

## Common Workflows

1. **Mathematik ändern:** `python -m pytest tests -q` (8 Tests) — muss grün
   bleiben, ohne Modell.
2. **Ein Test:** `python -m pytest tests/test_core.py::test_loo_axis_recovers_signal -q`.
3. **Achsen kalibrieren:** `provenance-audit calibrate --backend hf --model <id> --device cpu`
   schreibt `calibration.json` (README.md:48-52).
4. **Text prüfen:** `provenance-audit flag --backend hf --model <id> --text "…"`
   → `{"justified": …, "reason": …}`.
5. **Behauptungen auditieren:** `provenance-audit audit --backend hf --model <id> --text … --entities …`
   (Paraphrasen-Stabilität).
6. **Neues Backend:** Adapter unter `provenance_audit/adapters/` implementieren
   (Vorbilder: `hf.py`, `nanogpt.py`), dann `--backend` in `cli.py` ergänzen.

## Architecture Overview

1. **`core.py`** — Modellfreie Mathematik: Richtungen aus gepaarten
   Differenzen, Leave-one-out-Achsen, Permutationstests, AUROC, LDA.
2. **`cells.py`** — Minimalpaare, bei denen nur die Quelle der Antwort
   variiert (Kontextziffer, F+/F−, Modell-/Nutzertest).
3. **`axes.py`** — die drei Achsen als lineare Richtung im Residualstrom
   plus datengetriebene Schwelle.
4. **`indicator.py`** — Ein-Pass-Indikator: gerechtfertigt, wenn Kontext-
   **oder** Familiaritätsachse es deckt, sonst `unbacked`.
5. **`adapters/`** — `hf.py` (transformers) und `nanogpt.py` liefern nur
   Hidden States; read-only.
6. **`cli.py`** — `calibrate` / `flag` / `audit`, JSON-Ausgabe.

| Datei | Rolle |
| :--- | :--- |
| `provenance_audit/core.py` | Messmathematik (modellfrei) |
| `provenance_audit/cells.py` | Minimalpaar-Konstruktion |
| `provenance_audit/axes.py` | drei Provenienz-Achsen |
| `provenance_audit/indicator.py` | Ein-Pass-Halluzinationsindikator |
| `provenance_audit/adapters/hf.py` | transformers-Backend |
| `provenance_audit/adapters/nanogpt.py` | nanoGPT-Backend |
| `provenance_audit/cli.py` | CLI `calibrate`/`flag`/`audit` |
| `docs/` | METHOD, METRICS, PORTS |
| `results/` | eingefrorene Zahlen (`summary.json`) |

## Testing

`pytest`, 8 Tests, alle modellfrei (`tests/test_core.py`).

- **Alles:** `python -m pytest tests -q` → 8 passed (< 1 s).
- **Einzelfall:** `python -m pytest tests/test_core.py::test_auroc_known_values -q`.
- **Abgedeckt:** Null-Richtung, gepaarte Richtung, Schwellen-Mittelpunkt,
  LOO-Signal, LOO-Null nicht signifikant, Permutation ≠ Identität, AUROC,
  LDA-Scores.
- **Nicht abgedeckt:** Backends (brauchen Modell), CLI-Pfade mit echten
  Gewichten, die eingefrorenen `results/`-Zahlen.

## CI

Keine. Kein `.github/`, keine `.gitlab-ci.yml`, kein `tox`/`nox`
(verifiziert 2026-11-19). Der Vertrag ist der lokale `pytest`-Lauf; die
Konsumenten (z. B. `tiny21-provenance`) pinnen das Paket per Git-Referenz
und testen dort.

## Runtime Configuration / Usage

- **CLI:** `provenance-audit {calibrate,flag,audit}` mit
  `--backend {hf,nanogpt}`, `--model`, `--ckpt`, `--nanogpt-dir`, `--device`,
  `--dtype`, `--local`, `--calib`, `--text`, `--entities`, `--out`
  (`--help` verifiziert 2026-11-19).
- **Kalibrierung:** wird als JSON geschrieben und beim `flag`/`audit` über
  `--calib` gelesen.
- **`--local`:** verhindert Netzwerkzugriffe (Offline-Betrieb).
- **Voraussetzung:** ein Backend; ohne `.[hf]`/`.[nanogpt]` bleibt nur die
  modellfreie Mathematik nutzbar.

## Dependencies

- **Kern:** `numpy>=1.23` (einzige Pflicht-Abhängigkeit, `pyproject.toml`).
- **Extras:** `hf` → `torch`, `transformers`; `nanogpt` → `torch`,
  `tiktoken`; `test` → `pytest`.
- **Python:** `requires-python >= 3.9`.
- Keine vendorten Bibliotheken, keine Systembibliotheken.

## Source Layout

```
provenance_audit/        Paket: core, cells, axes, indicator, cli
provenance_audit/adapters/  hf.py, nanogpt.py (nur Hidden States)
docs/                    METHOD.md, METRICS.md, PORTS.md
results/                 summary.json + Erklärung (eingefroren)
tests/                   test_core.py (8 modellfreie Tests)
```

Build-Output: `pip install -e .` schreibt nur Metadata; es gibt kein
Artefakt im Repo.

## Generated artefacts (never hand-edit)

| Datei | Quelle | Kommando |
| :--- | :--- | :--- |
| `results/summary.json` | Entwicklungs-Geräte (CPU, float32) | kein Generator im Repo — bewusst eingefroren, nur mit neuer Messung ersetzen (`results/README.md`) |

`calibration.json` wird zur Laufzeit vom `calibrate`-Befehl geschrieben und
ist nicht eingecheckt.

## Writing pull requests / issues

Kein `CONTRIBUTING.md`, keine Templates (verifiziert 2026-11-19). Gelebte
Norm: kleine, fokussierte Commits im `bereich: gegenstand`-Stil; Änderungen
an der Mathematik nur mit Test, der die Aussage stützt. Die drei Achsen sind
ein Messversprechen — Erweiterungen müssen in `docs/METRICS.md` als
messbar/nicht messbar eingeordnet werden.

## Verification log

| Command | Status | Evidence | Date |
| :--- | :--- | :--- | :--- |
| `pip install git+https://github.com/bartman081523/provenance-audit.git` | verified | exit 0, frisches Venv | 2026-11-19 |
| `python -m pytest tests -q` | verified | 8 passed | 2026-11-19 |
| `provenance-audit --help` | verified | usage, Subcommands sichtbar | 2026-11-19 |
| `pip install -e ".[hf]"` | UNVERIFIED | pyproject.toml extras | 2026-11-19 |
| `provenance-audit calibrate --backend hf …` | UNVERIFIED | README.md:48-52, lädt ein Modell | 2026-11-19 |
