# Submission Bar Policy

## Core Principle

**Baseline ≠ Submission Candidate**

Baseline 只是起点，不是终点。

## Submission Criteria

### ❌ NOT Ready
- First baseline run (even if score is good)
- Simple probe/linear model
- No iteration history
- Family name contains: baseline, debug, smoke, probe

### ✅ Ready
- ≥5 rounds of iteration
- Applied techniques: ensemble, calibration, post-processing, domain adaptation
- Validated in knowledge/01_validated_findings.md
- Clear improvement trajectory in reports/

## Example

**Bad**:
- "Perch baseline" scores 0.75 → submit ❌
- Why: It's just cached embeddings + linear probe

**Good**:
- Round 1: "Perch baseline" → iterate
- Round 2: "Perch + calibration" → iterate
- Round 3: "Perch + ensemble" → iterate
- Round 4: "Perch + soundscape prior" → iterate
- Round 5: "Perch + post-processing" → iterate
- Round 6: "Perch full pipeline" → submit ✅
- Why: 6 rounds, multiple techniques, validated approach

## Implementation

Update `prompts/decider.md` to enforce this bar.
