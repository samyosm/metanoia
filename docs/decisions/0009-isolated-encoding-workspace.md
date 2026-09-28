# 0009 — Make the rounds necessary: isolated sentence encoding, workspace-only readout

**Evidence (R012).**
- The residual workspace of decision 0008 was trained on top of a full-context,
  fine-tuned direct path. It stayed switched off:
  - its readout grew to ‖w‖ ≈ 0.07, against direct logit spreads of 21–32;
  - answers were identical from 0 to 16 rounds.
- The direct path fit the training depths by itself, so the rounds received no
  gradient.
- A mechanism can only learn to use budgeted computation if that computation is needed
  on the training data.

**Choice.** Two new workspace options. The defaults keep decision 0008 unchanged.

- **`encoding = "isolated"`.** Each sentence of the rendered input (the question line,
  every fact and rule) is encoded **on its own** and pooled to one node. Each candidate
  is encoded from its own text only. The encoder therefore cannot combine premises:
  any link between "Kalo is dax", "Everything that is dax is also wug" and "Which one is
  wug?" must be made by workspace rounds. A depth-d question needs information to cross
  about d + 1 sentence boundaries.
- **`readout = "workspace"`.** Logits come only from the candidate nodes' states,
  through the readout head, with a normal (not zero) initialization. There is no
  direct-path bypass.
  - Budget 0 is the readout before any round: the model's honest no-deliberation
    answer, expected to be near chance on rule questions (PLAN §6.3, as revised).
  - `isolated` requires `readout = "workspace"`, since a full-context direct path would
    reintroduce the bypass.

**Training.**
- Warm start from R005, whose encoder is loaded; its scorer is loaded but unused.
- Rounds are sampled from 1–6, since depth 3 needs about 4. Deep supervision over
  every round.
- The same data (`microworld_v1_train20k`), schedule and step count as the R012
  baseline.
- The latent control keeps the same isolated encoding. Its slots read the isolated
  sentence nodes as fixed memory, so the difference is updated statements against
  fixed statements.

**What would count** (`reasoning_v1` revision 2, H1):
- Accuracy and NLL at held-out depths 5–8 improve with rounds, including past the
  trained range (r8, r16).
- The statement workspace beats the latent control at matched rounds.
- At some budget it beats the converged direct baseline of R012 at depths 5–8.

A result where rounds help but the direct baseline stays better is still informative:
the mechanism works but is weaker than a large single-pass encoder.

**Cost.** Isolated encoding runs about 47 short sequences (about 14 tokens each) plus 4
candidate names per example. That is fewer tokens than 5 joint sequences of about 480,
so training should be no slower.

**Revisit when** the isolated workspace cannot fit the training depths (then more
capacity per round, for example two layers, or explicit typed edges), or when it fits
them but rounds beyond training do not help (then halting, or step-wise supervision).
