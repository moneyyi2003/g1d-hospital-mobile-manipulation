import unittest

from hospital_vln.manipulation_scene import build_manipulation_scene


class ManipulationSceneTests(unittest.TestCase):
    def test_cube_rests_on_front_of_tabletop(self):
        scene = build_manipulation_scene((-2.5, 0.2, 1.05), 0.2)
        tabletop = scene.table_parts[0]

        self.assertEqual(tabletop.name, "Tabletop")
        self.assertEqual(len(scene.table_parts), 5)
        self.assertAlmostEqual(scene.tabletop_surface_z, 0.95)
        self.assertAlmostEqual(
            scene.cube.center[2] - scene.cube.size[2] / 2.0,
            scene.tabletop_surface_z,
        )
        table_front_y = tabletop.center[1] - tabletop.size[1] / 2.0
        self.assertAlmostEqual(scene.cube.center[1] - table_front_y, 0.10)

    def test_layout_has_positive_physical_dimensions(self):
        scene = build_manipulation_scene((-2.5, 0.2, 1.05), 0.2)

        self.assertGreater(scene.cube_mass_kg, 0.0)
        for part in scene.table_parts:
            self.assertTrue(all(value > 0.0 for value in part.size))

    def test_rejects_invalid_scene_parameters(self):
        with self.assertRaises(ValueError):
            build_manipulation_scene((0.0, 0.0, 1.0), 0.01)
        with self.assertRaises(ValueError):
            build_manipulation_scene((0.0, 0.0, 1.0), 0.1, cube_mass_kg=0.0)


if __name__ == "__main__":
    unittest.main()
