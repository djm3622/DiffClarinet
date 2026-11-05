"""Prototype searches respect the budget and find an easy pair."""

from itertools import product
import unittest

import torch
from torch import nn

from diagnostics import search_prototypes as p
from diagnostics.random_search import fit_random_search, propose
from model.kps.training import _synth
from scripts.run_search_prototypes import SETTINGS
from tests.test_filtered_fitting import make_source


def cfg(**extra) -> dict:
    return {**SETTINGS, "variant": "local", "samples": 4, "epochs": 60, "refine_epochs": 3, "n_fft": 32,
            "continuous_lr": .01, "discrete_lr": .003, "spectrum_normalization": "none", "log_every": 5,
            "log_calls": 50, "local_radius": 1, "local_probability": .5, "prior_start": .05,
            "prior_end": .001, "prior_anneal_epochs": 60, "smoothness": 1e-4, "advantage_clip": 5.0,
            "initial_K": .8, "initial_a": .2, **extra}


class PrototypeTests(unittest.TestCase):
    def test_swap_is_off_by_default_and_keeps_valid_pairs(self) -> None:
        models = [make_source("dp"), make_source("dp")]
        grids = [m.all_pairs() for m in models]
        valid = [set(g) for g in grids]
        incumbent = [grids[0][0], grids[1][-1]]
        settings = cfg(swap_probability=1.0)
        torch.manual_seed(0)
        for _ in range(50):
            pairs = propose(grids, valid, incumbent, settings)
            self.assertTrue(all(pair in valid[i] for i, pair in enumerate(pairs)))
        self.assertNotIn("swap_probability", dict(variant="local"))

    def test_all_prototypes_run_within_budget(self) -> None:
        fits = {"kprofile": lambda m, t, c, pos: p.fit_rs_kprofile(m, t, c),
                "elitist": lambda m, t, c, pos: p.fit_reinforce_elitist(m, t, c, pos),
                "rkprofile": lambda m, t, c, pos: p.fit_reinforce_kprofile(m, t, c, pos),
                "mcmc": lambda m, t, c, pos: p.fit_mcmc_pt(m, t, c),
                "swap": lambda m, t, c, pos: fit_random_search(m, t, c),
                "em": lambda m, t, c, pos: p.fit_em_backfit(m, t, c),
                "em_seeded": lambda m, t, c, pos: p.fit_em_backfit(m, t, c, init_L=[5]),
                "cem": lambda m, t, c, pos: p.fit_cem_joint(m, t, c),
                "mp": lambda m, t, c, pos: p.fit_mp_init(m, t, c),
                "gibbs": lambda m, t, c, pos: p.fit_gibbs_pt(m, t, c)}
        for name, position, count in product(fits, ("dp", "A"), (1, 2)):
            with self.subTest(name=name, position=position, count=count):
                torch.manual_seed(1)
                models = nn.ModuleList(make_source(position) for _ in range(count))
                target = _synth(models, 32, [(5 + i, 2) for i in range(count)]).detach()
                settings = cfg()
                history, calls = fits[name](models, target, settings, position)
                # One overshoot of a few syntheses is allowed at the budget boundary.
                limit = (settings["epochs"] * settings["samples"] + settings["refine_epochs"]) * count
                self.assertLessEqual(calls, limit + 8 * count * (1 + len(settings["K_grid"])))
                self.assertTrue(history)
                for model in models:
                    self.assertTrue(torch.isfinite(model.scaled_gain()))

    def test_em_backfit_recovers_an_easy_mixture(self) -> None:
        torch.manual_seed(2)
        models = nn.ModuleList(make_source("dp") for _ in range(2))
        truth = [(5, 2), (6, 3)]
        target = _synth(models, 32, truth).detach()
        history, _ = p.fit_em_backfit(models, target, cfg(epochs=200, em_top=4, em_refine_steps=2,
                                                          em_K_coarse=[.8], em_K_grid=[.8]))
        self.assertEqual(sorted(tuple(pair) for pair in
                                next(r for r in reversed(history) if "pairs" in r)["pairs"]), truth)


if __name__ == "__main__":
    unittest.main()
