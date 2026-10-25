"""Small controls for the matched-budget random-search baseline."""

from itertools import product
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diagnostics.random_search import fit_random_search, propose
from model.kps.training import _synth
from tests.test_filtered_fitting import make_source


def config(variant: str) -> dict:
    return dict(variant=variant, samples=4, epochs=30, refine_epochs=2, n_fft=32,
                continuous_lr=.01, spectrum_normalization="none", log_every=1,
                local_radius=1, local_probability=.5)


class RandomSearchTests(unittest.TestCase):
    def test_local_proposals_change_one_source_and_stay_valid(self) -> None:
        models = nn.ModuleList([make_source("dp"), make_source("dp")])
        grids = [model.all_pairs() for model in models]
        valid = [set(grid) for grid in grids]
        incumbent = [grids[0][0], grids[1][-1]]
        torch.manual_seed(0)
        for _ in range(50):
            pairs = propose(grids, valid, incumbent, config("local"))
            self.assertLessEqual(sum(a != b for a, b in zip(pairs, incumbent)), 1)
            self.assertTrue(all(pair in valid[i] for i, pair in enumerate(pairs)))

    def test_budget_selection_and_finite_updates(self) -> None:
        for position, count, variant in product(("dp", "A"), (1, 2), ("global", "local")):
            with self.subTest(position=position, sources=count, variant=variant):
                torch.manual_seed(5)
                models = nn.ModuleList(make_source(position) for _ in range(count))
                truth = [(5 + i, 2) for i in range(count)]
                target = _synth(models, 32, truth).detach()
                cfg = config(variant)
                with patch("diagnostics.random_search._synth", wraps=_synth) as synth:
                    history, calls = fit_random_search(models, target, cfg)
                self.assertEqual(synth.call_count, 30 * 4 + 2)
                self.assertEqual(calls, (30 * 4 + 2) * count)
                self.assertEqual(len(history), 32)
                fit_losses = [row["loss"] for row in history if row["phase"] == "fit"]
                self.assertTrue(torch.isfinite(torch.tensor(fit_losses)).all())
                for model, pair in zip(models, history[-1]["pairs"]):
                    self.assertEqual(model.selected_delays(), tuple(pair))
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
