# Required Questions — Answers

(See chat for the full detailed write-up this file mirrors; kept here so
it's directly available when assembling reports/report.pdf.)

## Q1. Which description is more useful, and which is more trustworthy, the
direct VLM description (Task 1) or the numbers-first description (Task 2),
and why?

Usefulness and trustworthiness split along different axes. The optimised
VLM description (Task 1) is more useful as a standalone artifact -- it
captures qualitative visual character (spatial arrangement, general
gestalt) a numbers-first summary cannot, and correctly identified modality
("fluorescence") while honestly flagging tissue_type as "uncertain." But
the naive VLM prompt, with no constraints, confidently claimed cells were
"actively dividing" and in a "state of rapid growth or proliferation" --
fabricated content with zero grounding, not determinable from a static
image.

The numbers-first description (Task 2) is reliable only for direct
pass-through fields: n_objects=22 was correctly copied from the Otsu
summary. But density_class="dense" was wrong (ground truth: "normal"),
because the model had no population-level reference to calibrate a single
image's numbers against. The narrative also mis-explained what "extent"
measures (claimed it indicates size variation; it measures shape
compactness) -- a logic error layered on a correctly-cited number.

Verdict: the VLM description is more useful when properly constrained;
the numbers-first approach is more trustworthy, but only for its literal
numeric fields -- its categorical judgments are no more reliable than the
VLM's, and sometimes worse. The clearest differentiator is auditability:
n_objects/mean_area can be checked against the regionprops table directly;
a VLM's claim about cell division state cannot be checked against
anything.

## Q2. Did the U-Net improve on classical Otsu segmentation for your
modality? Give one example image where each approach did better.

Two-part answer. Object counting: no improvement, effectively tied (mean
abs error Otsu=12.25, U-Net=12.58) -- because ground-truth BINARY masks
merge touching nuclei themselves, so both methods share the same
structural ceiling; this is not a modelling failure, it needs
watershed/instance segmentation to fix (matches the course lecture's
"Common Failure Modes" material).

Pixel-level Dice/IoU (the metric each model actually optimises): real,
modest U-Net improvement -- mean Dice 0.9815 (U-Net) vs 0.9784 (Otsu),
U-Net wins on 10/12 test images.

- U-Net did better: test_005 (clustered, 49 nuclei) -- Otsu Dice 0.978,
  U-Net Dice 0.989 (largest margin across the test set). Most
  touching/overlapping-nuclei image; U-Net's spatial context handles
  ambiguous boundaries better than Otsu's per-pixel threshold.
- Otsu did better: test_003 (normal, 21 nuclei) -- Otsu Dice 0.981, U-Net
  Dice 0.976. Well-separated, high-contrast image where a simple
  threshold was already near-ideal; margin is small.

## Q3. Report your U-Net's Dice and IoU. What do these numbers mean, and
where does the model tend to make its mistakes?

Final validation Dice = 0.9828, IoU = 0.9662 (20 held-out images).
Dice = 2|intersection|/(|pred|+|GT|); IoU = |intersection|/|union|
(stricter, IoU = Dice/(2-Dice) for the same masks). Both exceed the course
lab's own stated benchmark (0.85-0.95).

Across images: weakest scores (test_001, test_003, test_006, test_011,
0.973-0.976) don't track density simply -- test_006 is sparse (8 nuclei)
yet among the weaker cases, while the two densest images score well
(0.989, 0.979). Errors aren't simply "worse as density increases."

Within images (visual inspection, not a formal quantitative breakdown):
errors concentrate at object boundaries (CNN upsampling produces smoother,
more rounded edges than the angular ground-truth ellipses), which
disproportionately hurts small objects (a 1-2px boundary shift is a large
fraction of a small object's area). On touching clusters, prediction and
ground truth already agree on the merged blob's extent (since GT merges
them too), so residual error there is about the outer boundary, not an
internal division neither model attempts. A rigorous answer would bin
errors by object size/boundary-vs-interior explicitly -- not done here,
a natural extension.

## Q4. Where in the pipeline can the LLM hallucinate, and what design
choices reduce that risk? Why does keeping the structured JSON as the
"source of truth" help?

Four distinct hallucination modes found in this project's own evidence:
1. Fabrication ungrounded in anything real (Task 1 naive: "actively
   dividing," "rapid growth or proliferation").
2. Run-to-run instability (Task 1 optimised prompt, 3 identical calls,
   temp=0.8: modality confidence and image_quality both flip-flopped on
   the same static image).
3. Miscalibrated categorical judgment (Task 2: density_class wrong; Task
   4: density_class="moderate" on ALL 12 test images despite n_objects
   ranging 8-42 -- no population reference was ever given).
4. Incorrect reasoning about a correctly-cited fact (Task 2 narrative:
   correctly quotes extent=0.71, then wrongly explains what it means).

Design choices used here that reduce (not eliminate) this: explicit
descriptive-not-diagnostic anchoring; explicit permission for
"uncertain" (under-used in practice); numbers-only input for Tasks 2/4
(changes failure from unfalsifiable perception-hallucination to checkable
reasoning-hallucination); fixed schema with enumerated options (bounds
which wrong answers are even possible); temperature=0 for real pipeline
outputs (reproducibility).

Why JSON-as-source-of-truth helps: it separates output into two tiers
with honestly different reliability -- numeric pass-through fields
(auditable against the regionprops table) vs categorical/narrative
content (shown here to be unreliable). Task 4's CSV is built from JSON
records, not parsed narratives, so a user working from n_objects/mean_area
inherits real numeric reliability, while a user trusting density_class (or
the prose) would be working with data that's uniformly wrong in a way
that's invisible without checking back against the source numbers.

## Q5. Considering accuracy, auditability, and the limits of your dataset,
would you trust any part of this system in a real clinical setting? What
single change would most improve trustworthiness?

No, for three compounding reasons: (1) the dataset is fully synthetic
(idealised ellipses, no real staining artefacts/autofluorescence/overlap)
and small -- any accuracy figure describes this synthetic distribution
only, not real tissue; (2) the segmentation cannot separate touching
nuclei into correct counts (errors up to 42 objects), which most real
clinical use cases (proliferation indices, density-based diagnostics)
would depend on; (3) every LLM stage has a directly demonstrated,
reproducible failure in this project's own evidence, not hypothetical
risk -- matching the assignment's own "not cleared for clinical use"
framing concretely rather than just formally.

Single most impactful change: retrain segmentation as instance-aware (or
add marker-based watershed), using the instance-label masks the dataset
already provides -- chosen over LLM calibration-context fixes because the
counting failure is the most severe, most consistently demonstrated
problem (dozens of objects of error vs ~0.01 Dice differences), affects
BOTH segmentation methods equally (not fixable by model choice alone), and
is the most load-bearing number in the pipeline -- n_objects feeds every
downstream summary, LLM interpretation, and the final CSV. Fixing it
improves every downstream output at once; fixing only LLM calibration
would leave the upstream counting error fully intact underneath it.
