"""使用实际 MuJoCo 模型验证起摆、串级稳摆和失败判定。"""

import unittest

import mujoco
import numpy as np

from cart_pole.swing_up_pid import (
    Config, LimitedPID, SwingUpCascadePID, XML_PATH, simulate, upright_error,
)


class SwingUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.history, cls.summary = simulate(cls.config)

    def test_swing_up_reaches_and_holds_balance_without_hitting_rail(self):
        h, s = self.history, self.summary
        self.assertTrue(s["stable"], s)
        self.assertEqual(s["termination"], "completed")
        self.assertGreater(s["first_capture_s"], 0)
        self.assertLess(s["first_capture_s"], 5.0)
        self.assertEqual(len(s["mode_switches"]), 1)
        self.assertLess(s["max_abs_position_m"], 1.18)
        self.assertLessEqual(s["max_abs_force_n"], 20.0)
        for name in ("x", "theta", "x_dot", "theta_dot", "force", "energy"):
            self.assertTrue(np.all(np.isfinite(h[name])), name)
        tail = h["time"] >= h["time"][-1] - self.config.stable_duration
        self.assertTrue(np.all(h["mode"][tail] == 1))
        self.assertTrue(np.all(np.abs(h["theta"][tail]) < self.config.stable_angle))
        self.assertTrue(np.all(np.abs(h["x"][tail]) < self.config.stable_position))
        self.assertTrue(np.all(np.abs(h["x_dot"][tail]) < self.config.stable_cart_speed))
        self.assertTrue(np.all(np.abs(h["theta_dot"][tail]) < self.config.stable_pole_speed))

    def test_cascade_recovers_both_sides_of_upright(self):
        for initial in (175, 185):
            with self.subTest(initial_angle=initial):
                _, summary = simulate(self.config, initial_angle=np.deg2rad(initial))
                self.assertTrue(summary["stable"], summary)
                self.assertEqual(summary["first_capture_s"], 0.0)
                self.assertLess(abs(summary["final_position_m"]), 0.002)

    def test_lost_balance_resets_integrals_and_returns_to_swing_up(self):
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        controller = SwingUpCascadePID(model, self.config)
        controller.update(0, 0, 0, 0, 0.002, 1.0)
        self.assertEqual(controller.mode, "balance")
        controller.position_pid.integral = 0.5
        controller.angle_pid.integral = 0.5
        force = controller.update(0, np.deg2rad(40), 0, 0, 0.002, 1.002)
        self.assertEqual(controller.mode, "swing_up")
        self.assertEqual(controller.position_pid.integral, 0.0)
        self.assertEqual(controller.angle_pid.integral, 0.0)
        self.assertTrue(np.isfinite(force))
        self.assertTrue(np.isnan(controller.theta_ref))

    def test_rail_approach_is_reported_as_failure(self):
        _, summary = simulate(Config(duration=1.0), initial_x=1.17)
        self.assertFalse(summary["stable"])
        self.assertEqual(summary["termination"], "rail_margin_reached")
        self.assertLess(summary["duration_s"], 1.0)

    def test_short_run_is_not_reported_as_stable(self):
        _, summary = simulate(Config(duration=0.1), initial_angle=np.pi)
        self.assertFalse(summary["stable"])


class PIDTests(unittest.TestCase):
    def test_saturation_does_not_accumulate_integral(self):
        pid = LimitedPID((1.0, 1.0, 0.0), 2.0)
        for _ in range(100):
            self.assertEqual(pid.update(3.0, 0.0, 0.01), 2.0)
        self.assertEqual(pid.integral, 0.0)
        self.assertLess(pid.update(0.1, 0.0, 0.01), 0.2)

    def test_reverse_error_can_unwind_integral(self):
        pid = LimitedPID((0.0, 1.0, 0.0), 1.0)
        pid.integral = 1.0
        self.assertLess(pid.update(-0.5, 0.0, 0.1), 1.0)
        self.assertLess(pid.integral, 1.0)

    def test_angle_convention_and_periodicity(self):
        self.assertAlmostEqual(abs(upright_error(0)), np.pi)
        for q in (np.pi, -np.pi, 3 * np.pi):
            self.assertAlmostEqual(upright_error(q), 0.0)


if __name__ == "__main__":
    unittest.main()
