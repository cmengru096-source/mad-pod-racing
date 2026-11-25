import unittest
from unittest.mock import patch

import search_demo as demo
from physics import Pod


class SearchDemoTests(unittest.TestCase):
    def test_fixed_budget_and_seed(self):
        case = demo.CASES[0]
        pods = [Pod(*case.start), Pod(*case.traffic)]
        first, second = demo.Evolution(), demo.Evolution()
        with patch.object(demo, '_evaluate', wraps=demo._evaluate) as evaluate:
            decision = first(pods, case.track)
        self.assertEqual(evaluate.call_count, first.budget)
        self.assertEqual(decision, second(pods, case.track))
        self.assertEqual(first.best, second.best)
        self.assertEqual(pods, [Pod(*case.start), Pod(*case.traffic)])

    def test_head_on_shield_improves_distance_and_finish(self):
        case = demo.CASES[0]
        search = demo.run(case, demo.Evolution())
        baseline = demo.run(case, demo.greedy)
        self.assertEqual(search.shields, (1,))
        self.assertEqual(search.passed, len(case.track))
        self.assertEqual(baseline.passed, len(case.track))
        self.assertLess(search.remaining_at_12, baseline.remaining_at_12)
        self.assertLess(search.turns, baseline.turns)

    def test_clear_circuit_compares_full_laps(self):
        case = demo.CASES[1]
        search = demo.run(case, demo.Evolution())
        baseline = demo.run(case, demo.greedy)
        self.assertEqual(search.shields, ())
        self.assertEqual((search.remaining, baseline.remaining), (0.0, 0.0))
        self.assertEqual((search.passed, baseline.passed), (4, 4))
        self.assertLess(search.turns, baseline.turns)


if __name__ == '__main__':
    unittest.main()
