import math
import unittest

from physics import Command, Pod, collision_time, step


COAST = Command(15000, 0, 0)
FAR_GATE = [(15000, 8000), (1000, 8000)]


def pod(x=0, vx=0, y=0, vy=0):
    return Pod(x, y, vx, vy, 0, 0)


class PhysicsTests(unittest.TestCase):
    def test_contact_time_and_horizon(self):
        a, b = pod(0, 300), pod(1100, -300)
        self.assertEqual(collision_time(a, b), 0.5)
        self.assertIsNone(collision_time(a, b, 0.499))
        b.vx = 300
        self.assertIsNone(collision_time(a, b))
        b.vx = 500
        self.assertIsNone(collision_time(a, b))

    def test_equal_mass_head_on(self):
        a, b = pod(0, 300), pod(1100, -300)
        step([a, b], [COAST, COAST], FAR_GATE)
        self.assertEqual((a.x, b.x, a.vx, b.vx), (0, 1100, -255, 255))

    def test_minimum_impulse_at_low_speed(self):
        a, b = pod(0, 20), pod(810)
        step([a, b], [COAST, COAST], FAR_GATE)
        self.assertEqual((a.x, b.x, a.vx, b.vx), (-45, 875, -93, 110))

    def test_shield_mass_only_on_activation(self):
        a, b = pod(0, 300), pod(1100, -300)
        step([a, b], [Command(10000, 0, 'SHIELD'), COAST], FAR_GATE)
        self.assertEqual((a.vx, b.vx, a.shield), (162, 672, 3))
        a.x, a.vx, b.x, b.vx = 0, 300, 1100, -300
        step([a, b], [Command(10000, 0, 100), COAST], FAR_GATE)
        self.assertEqual((a.vx, b.vx, a.shield), (-255, 255, 2))

    def test_shield_engine_delay_and_repeated_boost(self):
        a = pod()
        for power, expected_vx in [('SHIELD', 0), (100, 0), (100, 0), (100, 0), ('BOOST', 552), ('BOOST', 554)]:
            step([a], [Command(10000, 0, power)], FAR_GATE)
            self.assertEqual(a.vx, expected_vx)
        self.assertFalse(a.boost)

    def test_first_turn_and_limited_rotation(self):
        a = pod()
        a.angle = -1
        step([a], [Command(0, -1000, 100)], FAR_GATE)
        self.assertEqual((a.angle, a.y, a.vy), (270, -100, -85))
        step([a], [Command(10000, -100, 100)], FAR_GATE)
        self.assertAlmostEqual(a.angle, 288)

    def test_gold_boost_is_shared_only_between_teammates(self):
        pods = [pod(y=i * 2000) for i in range(4)]
        boosts = [Command(10000, p.y, 'BOOST') for p in pods]
        coast = [Command(10000, p.y, 0) for p in pods]
        step(pods, [boosts[0], boosts[1], coast[2], coast[3]], [])
        self.assertEqual([p.vx for p in pods], [552, 85, 0, 0])
        self.assertEqual([p.boost for p in pods], [False, False, True, True])
        step(pods, [coast[0], boosts[1], coast[2], boosts[3]], [])
        self.assertEqual([p.vx for p in pods], [469, 157, 0, 552])
        self.assertEqual([p.boost for p in pods], [False] * 4)

    def test_two_opponents_each_get_a_boost_in_isolation(self):
        pods = [pod(), pod(y=2000)]
        step(pods, [Command(10000, p.y, 'BOOST') for p in pods], [])
        self.assertEqual([p.vx for p in pods], [552, 552])

    def test_spent_boost_cannot_reappear_from_teammate_state(self):
        pods = [pod(y=i * 2000) for i in range(4)]
        pods[1].boost = False
        step(pods, [Command(10000, p.y, 'BOOST' if i == 0 else 0)
                    for i, p in enumerate(pods)], [])
        self.assertEqual(pods[0].vx, 85)
        self.assertFalse(pods[0].boost)

    def test_swept_gate_crossings_in_order(self):
        a = pod(0, 4000)
        step([a], [COAST], [(1000, 0), (3000, 0), (1000, 5000)])
        self.assertEqual((a.passed, a.next_cp), (2, 2))
        b = pod(0, 4000, y=601)
        step([b], [COAST], [(1000, 0), (3000, 0)])
        self.assertEqual((b.passed, b.next_cp), (0, 0))

    def test_collision_diverts_pod_away_from_checkpoint(self):
        a, b = pod(0, 1800), pod(1200)
        step([a, b], [COAST, COAST], [(1800, 0), (9000, 0)])
        self.assertEqual((a.passed, b.passed), (0, 1))

    def test_two_contacts_in_one_turn(self):
        pods = [pod(0, 900), pod(1000), pod(2000), pod(5000)]
        step(pods, [COAST] * 4, FAR_GATE)
        self.assertEqual([p.vx for p in pods], [0, 0, 765, 0])

    def test_rounding_and_validation(self):
        a = pod(-4.5, -2)
        step([a], [COAST], FAR_GATE)
        self.assertEqual((a.x, a.vx), (-6, -1))
        with self.assertRaises(ValueError):
            Command(0, 0, 101)
        with self.assertRaises(ValueError):
            Command(math.inf, 0, 100)
        with self.assertRaises(ValueError):
            step([a], [], FAR_GATE)


if __name__ == '__main__':
    unittest.main()
