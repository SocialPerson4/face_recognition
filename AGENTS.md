# Project collaboration rules

## Research integrity

- Treat this repository as original research work, not a reproduction exercise.
- Python is the only formal implementation. MATLAB and old course-report artifacts stay outside the repository.
- Never invent, beautify, or selectively omit experimental results. A failed hypothesis is a valid result when the protocol and evidence are complete.
- Fit preprocessing, feature extraction, models, and thresholds using training data only. Keep final test data isolated from model and parameter decisions.
- Record every material change to the dataset, protocol, model, metric, or hypothesis in `docs/research_thinking.md` before interpreting its result.

## Explanation and documentation

- Explain each new concept in this order when useful: plain-language meaning, its role here, essential mathematics, code location, relevant course connection, and a short defense-ready answer.
- Keep `docs/code_guide.md` synchronized with code modules, functions, inputs, outputs, design reasons, and likely questions.
- Keep research decisions and the question log in `docs/research_thinking.md`.
- Keep stage speeches and personal interview preparation only in `local/interview_prep.md`. The entire `local/` directory must remain ignored by Git.

## Figures and presentations

- Generate figures from saved experiment data with reproducible scripts. Save source values and export a vector version plus a high-resolution PNG when appropriate.
- Treat the current PPT figure list as a narrative skeleton. Review every real figure with the researcher before deciding its conclusion, caption, layout, or inclusion in the final slides.
- Decorative images must not replace experimental evidence, and no caption may claim more than the data supports.

## Dependencies and workflow

- Lightweight routine Python dependencies may be installed directly and must be declared in `pyproject.toml`; document why each tool is used.
- Explain cost and obtain a decision before adding large models, GPU frameworks, system-level software, or dependencies with substantial disk/resource impact.
- At each stage: implement, test, save auditable outputs, update public documentation, update the local speech, commit, and push.
- Never commit raw face datasets, restricted data, local interview notes, caches, or temporary artifacts.
