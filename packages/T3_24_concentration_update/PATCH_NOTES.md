# T3.24 implementation notes

The experimental arm retains the existing no-reference raw-window residual network and additionally trains independent copies of the pretrained concentration mixture fusion and ordinal heads. The original CNN, encoder, peak-guided queries, projections, classifier and ordinal model remain frozen. This intentionally tests additional concentration adaptation, with a matched residual-only arm.

The frozen cache appends the pre-fusion [B,3,128] queries to the unchanged 1,266 residual features. The original 72 reference matching features remain masked. New concentration logits are expressed as their difference from the immutable original concentration path, evaluated using the same queries and batch shape, plus the original cached logits. This avoids turning floating-point differences from cache batch size into a nonzero initial change.

Copied concentration modules remain in eval mode, with requires_grad enabled. Only their dropout is disabled; Adam still updates their weights. The copied fusion has 131,971 parameters and copied heads 12,870. Both arms train the 40,742 residual parameters. The experimental model therefore has 185,583 trainable and 1,267,823 frozen parameters.

Both arms use lr=1e-5, minimum_lr=1e-7, three warmup epochs, 50 total epochs, batch16, Adam without weight decay, and gradient norm clipping at1. The objective is 0.7*CORN+0.02*total_delta_squared. No middle separation or preservation term is applied. The residual component retains its bound; the combined component has no additional hard bound.

Prediction CSVs separate residual_delta and branch_delta from their sum. Epoch histories report the maximum branch/residual/combined change and preclip gradient norm. Last-epoch parameter changes are explicitly labeled as last-epoch, not best-candidate changes. The adoption rule remains the original conservative T3.23 rule, with a zero-change original fallback and smoke ineligible for adoption.

Prepare verifies and snapshots project code without overwriting it. It checks historical report metrics and fingerprints T3.22 and T3.23 candidate/recommended files. Existing T3.18 and optional diagnosed T3.20 files remain protected. The package uses original lexical data paths so path-seeded runtime tail completion stays identical.

Checkpoint version: T3.24_concentration_update_v1. All inference dependencies are in this package plus its verified runtime Dataset snapshot. Loading restores copied trainable modules and saved buffers without reinitializing the trained branch or refitting references. Local verification uses artificial spectra and checkpoints only; it does not measure server accuracy.
