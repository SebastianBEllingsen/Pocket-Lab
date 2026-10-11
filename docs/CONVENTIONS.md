# Code Conventions and Design Rules

Agreed 2026-10-11. Goal: software that is easy to modify later, because we cannot know yet what will change.

## Layout
- One importable package at the project root, `cellcount/`, with a sub-package per pipeline stage. Each stage is runnable as `python -m cellcount.<stage>` (or the shorter module form agreed with Sebastian).
- Pure logic is separated from I/O and display:
  - `core` functions take arrays/dataclasses and return arrays/dataclasses. No printing, no file paths, no windows.
  - `cli.py` per stage parses flags and calls the core.
  - Visualization lives in a shared `cellcount/viz/` module, not inside the algorithms.
- Shared pieces (dataset loading, image selection, metrics, result saving, rating storage) live in `cellcount/common/` so every stage CLI behaves the same.

## Design rules
- Check open source first, then weigh build vs. adopt (Sebastian, 2026-10-11): this does not discourage building our own tools. If a better option exists, use or adapt it; if our own tool is better, build it. Compare the pros and cons of both each time (fit, accuracy, effort, licence, dependencies, GPU/size, control, course rules). Wrap them in a thin adapter that implements our own stage interface, so a framework can be swapped or compared against our own method. Candidate list: IDEAS.md section 8. Check licence and maintenance first.
- Algorithms are swappable behind a small interface (a function or class with a stable signature), selected by a flag (`--method`). New methods are added, not edited in.
- Parameters live in dataclasses with defaults (a config object), never as magic numbers buried in functions. Every parameter that could matter has a CLI flag.
- Datasets sit behind one loader interface (`Sample` with image, optional mask, optional boxes, name, dataset id). Paths are never hardcoded; the data root is a flag or env var.
- Deterministic by default: any randomness takes a `--seed`.
- Type hints on public functions, small functions, no global state.
- Tests for core logic (stdlib `unittest`, run with `python -m unittest discover -s tests -t .`), using tiny synthetic images where possible.

## File header rule
Every source file starts with a module docstring that contains: what the file is for, its public functions/classes in one line each, and, if it has a CLI, its flags with a one-line description each. Keep it readable for everyone on the project.

## CLI rules
- Options are flags, always. Shared flags behave identically in every stage:
  - Image selection: `--dataset`, `--image NAME_OR_PATH`, `--index N`, `--random N`, `--worst N` (lowest score), `--seed`.
  - Display: `--show` (window), `--save DIR`, plus a view-level flag (for example `--full` for the full comparison, `--minimal` for original vs. final).
- Visual output must carry enough information to understand what is going on (titles, parameter values, scores, image names); exact content agreed per stage.

## Version control
- Commit locally after each logical change, with a message that says what and why. One change per commit.
- Never push unless Sebastian explicitly asks. Data, training runs and weights are git-ignored and must never be added.

## Review loop
- Each stage ships a CLI, Sebastian looks at the output, and Sebastian's verdict is the sign-off. Verdicts are recorded in `docs/DECISIONS.md`.
- Planned: a rating tool where Sebastian marks outputs good/bad (or other parameters). Ratings are saved to disk and become part of the "golden" evaluation set.
