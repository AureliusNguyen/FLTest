# FLTest

### A Testbed for Enhancing Privacy and Robustness of Federated Learning Systems

FLTest is an open, community testbed for **evaluating the privacy and robustness of
Privacy-Preserving Federated Learning (PPFL)**. One YAML config runs the same experiment
across several FL frameworks, injects attacks and defenses as composable hooks, and applies
differential testing, metamorphic testing, and a pitfall checker. The goal is
software-defined control and visibility over how PPFL techniques are evaluated.

> Developed under the **NSF PDaSP Program (Track 3), Award #2452817-19**.

---

## The problem

A survey of 50 federated-learning robustness papers found experimental setups that vary
widely and flatter the technique under test. About 30% of those papers evaluate on MNIST
alone, roughly half use IID data, and about 40% rely on naive attacks such as random
Gaussian, label flipping, and sign flipping. Only 4% report per-client accuracy, so a single
global number hides the clients a defense leaves behind. These choices systematically
over-estimate privacy and robustness, and they make results hard to reproduce or compare.
No standard procedure exists for testing whether a PPFL technique is private and robust in
practice.

## What FLTest does

**Orchestration.** One configuration file drives end-to-end federated-learning experiments
across several frameworks through a single `run_simulation()` abstraction. The same setup
therefore runs unchanged on a dependency-light reference oracle, on Flower, and on NVFlare,
and the results are directly comparable.

**Hooks.** Attacks, defenses, and metric listeners are all hook plugins that share one
context. A plugin is written once and runs on every backend, and several plugins compose on
a single run. For example, `examples/configs/defense_robust.yaml` combines a backdoor attack
with median aggregation and reports clean accuracy alongside attack success rate.

**Testing.** Differential testing checks that frameworks given the same configuration agree
within tolerance, so an unexplained divergence points to an implementation bug. Metamorphic
testing checks relations that ought to hold, such as accuracy not dropping when the number
of IID clients doubles. The pitfall checker inspects a configuration against a catalog of
known evaluation pitfalls, and a recommendation engine turns each finding into a
counter-experiment.

**Fuzzing.** Any list-valued knob expands into a grid of runs, which covers models,
datasets, distributions, and attack settings from a compact config. For example, sweeping
`data_distribution` shows median aggregation cutting backdoor success from 0.80 to 0.03 on
IID data, while the same defense leaves it at 0.30 under a Dirichlet split with alpha 0.3.

## Project goals

The project designs, develops, and sustains FLTest as a standardized testbed that automates
privacy and robustness evaluation for PPFL systems. The testbed lowers the barrier to
rigorous federated-learning research for novice and expert users alike. Reproducible and
portable results come from cross-framework and cross-configuration comparison across Flower,
NVFlare, IBM FL, and other frameworks. Over-estimated privacy and robustness claims are
countered by detecting evaluation pitfalls and remediating them automatically. Advanced
attacks and defenses are first-class composable components, among them gradient-inversion
privacy attacks, backdoors, robust aggregation, and differential privacy. FLTest is built to
grow a sustainable open-source community around trustworthy FL evaluation, integrated with
existing FL frameworks and NSF cyberinfrastructure.

## Objectives

FLTest delivers an automated test-orchestration module that generates diverse,
fault-revealing federated deployments from a single configuration. A pitfall checker backs
that module with a catalog of known FL-evaluation pitfalls, updated as new research appears,
and a recommendation engine turns each detected pitfall into an actionable
counter-experiment. Privacy-preserving techniques such as differential privacy, secure
aggregation, and robust aggregation are supported alongside privacy- and
robustness-specific metrics. Those metrics include per-client evaluation, which exposes the
representation disparity that a single global number hides. The testbed is designed for
deployment at scale on FL frameworks and cloud testbeds, so that techniques are validated
under realistic conditions.

---

## Investigators

| Principal Investigator | Institution |
|------------------------|-------------|
| [**Ali Anwar**](https://chalianwar.github.io/) | University of Minnesota |
| [**Muhammad Ali Gulzar**](https://people.cs.vt.edu/~gulzar/) | Virginia Tech |
| [**Fatima Anwar**](https://people.umass.edu/fanwar/) | University of Massachusetts Amherst |

## Sponsor & partner institutions

<p align="center" style="display:flex; align-items:center; justify-content:center; gap:48px; flex-wrap:wrap;">
  <img src="assets/logos/nsf.png" alt="U.S. National Science Foundation" style="height:72px; width:auto;">
  <img src="assets/logos/umn.png" alt="University of Minnesota" style="height:48px; width:auto;">
  <img src="assets/logos/vt.png" alt="Virginia Tech" style="height:56px; width:auto;">
  <img src="assets/logos/umass.png" alt="University of Massachusetts Amherst" style="height:40px; width:auto;">
</p>

## Acknowledgement

This material is based upon work supported by the **U.S. National Science Foundation** under
the **Privacy-preserving Data Sharing in Practice (PDaSP) program, Track 3 — Usable Tools and
Testbeds for Confidential Data Sharing**, **Award #2452817-19**. The PDaSP program is supported
by the NSF together with its co-sponsors (U.S. Department of Transportation, Intel, NIST, and
Broadcom). Any opinions, findings, and conclusions or recommendations expressed in this
material are those of the authors and do not necessarily reflect the views of the National
Science Foundation or its co-sponsors.

Program information: [pdasp.net/projects](https://pdasp.net/projects/).

## Contact

Reach the investigators via their pages:

- [Ali Anwar](https://chalianwar.github.io/) — University of Minnesota
- [Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/) — Virginia Tech
- [Fatima Anwar](https://people.umass.edu/fanwar/) — University of Massachusetts Amherst

For bugs, feature requests, and contributions, please use the
[GitHub issue tracker](https://github.com/SEED-VT/FLTest/issues).

---

## Get started

<div class="grid cards" markdown>

- **[Install](installation.md)** — set up the isolated environment.
- **[Quickstart](quickstart.md)** — run your first experiment.
- **[Approach walkthrough](walkthrough.md)** — how a config becomes a tested result.
- **[Worked example](worked-example.md)** — exhaustively testing attacks & defenses.
- **[Configuration](configuration.md)** & **[Fuzzing](fuzzing.md)** — every knob, and the grid.
- **[Port your attacks & defenses](extending.md)** — bring your own technique.

</div>
