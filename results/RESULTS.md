# Results

BFCL accuracy on 5 categories (1,240 questions). Base is `Qwen/Qwen2.5-1.5B-Instruct`. Verified and Unverified are the same QLoRA recipe (r=16, 2 epochs, 10,000 rows) trained on data with and without the `verify.py` checks. Changes are in percentage points.

| Category | Base | Verified | Unverified | Verified vs Base | Unverified vs Base |
| --- | --- | --- | --- | --- | --- |
| `simple_python` | 86.50% | 87.50% | 89.25% | +1.0 | +2.8 |
| `multiple` | 84.00% | 87.00% | 88.50% | +3.0 | +4.5 |
| `parallel` | 74.50% | 78.00% | 81.00% | +3.5 | +6.5 |
| `parallel_multiple` | 72.50% | 75.50% | 77.50% | +3.0 | +5.0 |
| `irrelevance` | 72.50% | 57.92% | 55.00% | -14.6 | -17.5 |

## Are the differences real?

Each pair of models answered the same questions, so this counts the questions only one of the two got right and runs an exact McNemar test: the chance of a split at least this uneven if neither model were better. Below 0.05 is read as a real difference. Tool-calling and irrelevance questions are tested apart, because fine-tuning moved them in opposite directions. The test covers question-to-question luck only, not training luck: each model was trained once.

| Questions | Comparison | Only first right | Only second right | p-value | Verdict |
| --- | --- | --- | --- | --- | --- |
| tool calling | Verified vs Base | 61 | 38 | 0.027 | real difference |
| tool calling | Unverified vs Base | 76 | 33 | <0.001 | real difference |
| tool calling | Verified vs Unverified | 16 | 36 | 0.008 | real difference |
| irrelevance | Verified vs Base | 18 | 53 | <0.001 | real difference |
| irrelevance | Unverified vs Base | 18 | 60 | <0.001 | real difference |
| irrelevance | Verified vs Unverified | 12 | 5 | 0.143 | could be chance |

## Failure types

Counted automatically from BFCL's score files (see `evaluate.classify_failure`).

| Failure type | Base | Verified | Unverified |
| --- | --- | --- | --- |
| called a tool when it shouldn't | 66 | 101 | 108 |
| didn't call a tool when it should | 6 | 0 | 0 |
| bad format | 3 | 2 | 1 |
| wrong function | 5 | 7 | 6 |
| wrong number of calls | 34 | 30 | 19 |
| wrong arguments | 144 | 130 | 123 |
| other | 0 | 0 | 0 |
| **total** | 258 | 270 | 257 |

## Labelled failures

From 20 randomly chosen failures of the verified model (labelled by Claude (AI assistant), from reading each failure with its tools; not yet reviewed by hand):

- **near-miss tool**: 7
- **wrong argument value**: 4
- **wrong number of calls**: 3
- **dropped function-name prefix**: 2
- **multi-step maths not carried through**: 2
- **BFCL answer debatable**: 2

## Notes

**Did verification help?** No. On the same 1,000 tool-calling questions, the unverified model got 36 right that the verified model missed, against 16 the other way (p = 0.008), and the irrelevance difference (12 vs 5, p = 0.14) could be chance. One caveat: each model was trained once, so training luck is not ruled out (see below).

**What fine-tuning did.** Both fine-tuned models call tools better than the base model on all four tool-calling categories (+1.0 to +6.5 points, real in the paired test). Both got much worse at `irrelevance`, where the right answer is not to call a tool (−14.6 and −17.5 points, also real). The failure counts show the trade: "didn't call a tool when it should" went from 6 to 0, while "called a tool when it shouldn't" went from 66 to 101 and 108.

**Why irrelevance got worse.** The 1,000 no-tool training examples pair a request with clearly unrelated tools. BFCL's irrelevance questions are near misses: a tool on the same topic that can't actually answer, such as a route planner for "where is the nearest pharmacy". In the labelled sample, 7 of 9 irrelevance failures are near-miss calls like this. The other 2 look like questionable BFCL answers, where the given tool does seem to answer the question. The likely fix is near-miss no-tool examples in the training data.

**Why verification may not have helped.** These are hypotheses, not tested:
1. *Training luck.* The two training sets share all but 198 of 10,000 rows (2%). A 2-point gain from 2% of the data is surprising. Different data also changes the order of training batches, which by itself can move scores. The p-value covers which questions were asked, not this. Training each model again with a second seed (about 9 GPU hours for both) would separate the two.
2. *The verifier may reject useful rows.* It rejects rows whose argument types don't match the declared types, or that miss required parameters. Some of those may still teach the call shape BFCL rewards. The unverified model has fewer "wrong number of calls" failures (19 vs 30), which would fit this. Comparing the 198 rows that differ would test it.

**Run-to-run noise.** Re-running BFCL on the same model changed at most 1 question per category (baseline irrelevance 72.08% → 72.50%; verified parallel 78.50% → 78.00%), so evaluation noise is small next to these differences.
