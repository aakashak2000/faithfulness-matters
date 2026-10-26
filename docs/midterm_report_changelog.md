# Midterm report revision, change log (2026-10-25)
- Opening rewritten: concrete debugging scenario, plain-language task, one-paragraph finding (+20.8 vs -0.9), research questions aligned with the proposal (RQ1, RQ2 preliminary; RQ3 untested).
- Corrected-rule results are the primary findings (Table 2 first two rows, Figure 1); uncorrected-probability results are a labelled sensitivity block. New publication-style figure `figures/fig7_gain_corrected_rule.{png,pdf}` (script `scripts/48_figure_corrected_rule.py`, reads only the results CSVs).
- Related work is now a four-row comparison table (Liu et al., He and Xiao, Hessel and Lee, Jin et al.) plus a plain statement of what we add and do not claim.
- Method explains the class-prior correction and the swap test in plain English, each with a labelled made-up example; donor pool (8,004 training utterances, all used, mean 8,003 text and 8,004 audio valid donors per test utterance) and bootstrap documented.
- Models table adds M3 and states why M3 replaces M5 in the conflict experiment; weighted and macro F1 added (macro from the verified F1 recomputation).
- He and Xiao reference updated to the journal article (Circuits, Systems, and Signal Processing 45(9):7634-7652, 2026, DOI 10.1007/s00034-026-03547-w, checked through Crossref); quotations are from the arXiv v1 text and are marked as such.
- Audit-status prose, per-file paths and commit lists removed from the narrative; one reproducibility sentence kept. Status table separates "protocol frozen, annotation pending, no results" from completed work.
- Built from `docs/midterm_report_draft.md` with pandoc to `docs/midterm_report.tex`, compiled with xelatex to `docs/midterm_report.pdf` (4 pages, 10 pt, single spacing, 0.72 in margins); copied to `docs/midterm_report_draft.pdf`.

## Final corrections (2026-10-25)
1. Donor pool: 8,004 is correct for M0b to M5 and the corrected-rule runs (dialogue-level split v2; verified from `data/v2_pipeline.pt`, 8,004 clf rows, and the cached embeddings). 7,990 is the clf size of the ORIGINAL split used by the steps 04 to 20 headers. Notes fixed in `research_audit_2026-10-23.md` and `model_architecture_registry.md`; the report was already right.
2. Prior-correction example: frequencies now sum to one (0.50, 0.30, 0.20); scores 0.320, 0.311, 0.134 renormalise to 0.418, 0.407, 0.175 (checked numerically).
3. Related-work column renamed "Gate weights checked against interventions?"; Hessel and Lee and Jin et al. marked "n/a" with the reason (no gates; masking saliency used for training).
4. Commit hashes: 4f4c662 is the corrected-rule protocol; the conflict protocol was frozen in 5f878c7 and amended before annotation in ff55b48 (checked with git log).
5. Added one two-sentence hypothesis statement (H1 exploratory, H2 from the proposal).
Rebuilt `midterm_report.tex` and `midterm_report.pdf`: 4 pages, 10 pt class option, single spacing.
