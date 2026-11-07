# V3 GitHub release checklist

- [ ] Keep the frozen DLS/OSC baseline result (seed42000, n100, 84%/81%) intact.
- [ ] `smoke_sac_robosuite_exact.py` passes.
- [ ] `smoke_dls_osc.py --mode both --seed 42000 --steps 5` passes.
- [ ] `python -m unittest discover -s tests -p 'test_robustness_offline.py' -v` passes.
- [x] Robustness baseline n=100 reproduces DLS 84/100, OSC 81/100 before stress results are interpreted (per Windows console summary).
- [x] Console summary reports all seven scenarios n=100 per controller. [ ] Audit original episodes/design files for final reproducibility.
- [ ] Review `design.json`, `episodes.csv`, `summary.json`, model + checkpoint SHA-256 hashes.
- [ ] Make plots with `plot_robustness.py`; include verified plots in README after running experiments.
- [ ] Run `pip freeze` on tested Windows environment and include `requirements-lock.txt`.
- [ ] Bring model-specific upstream Menagerie `LICENSE` into publishable repo if assets are included.
- [ ] Decide license for own code and add top-level `LICENSE`.
- [ ] Review public tree; no weight ZIPs, private paths, trace dumps, venvs, credentials.
- [ ] Tag a release only after the real physics runs and README claims are verified.
