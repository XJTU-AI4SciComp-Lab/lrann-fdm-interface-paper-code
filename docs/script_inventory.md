# Paper Solver Inventory

Only the 15 solvers below are included. All correspond to a paper example, model column, or derivative/formulation comparison. Paths in the original-source column are relative to the author's initial code archive.

| Solver | Paper role | Original source |
| --- | --- | --- |
| [flower_interface_paper_final.py](../experiments/ex1_flower/ex1_case1/flower_interface_paper_final.py) | Example 4.1 Case 1; Tables 1-2; Figure 3 | `ex1_flower/ex1_case1/flower_interface_paper_final.py` |
| [mix_flower_interface_paper_final_2.py](../experiments/ex1_flower/ex1_case1/mix_flower_interface_paper_final_2.py) | Example 4.1 Case 1, mixed; Tables 3-4 | `ex1_flower/ex1_case1/mix_flower_interface_paper_final_2.py` |
| [circle_2d_1l.py](../experiments/ex1_flower/ex1_case2/circle_2d_1l.py) | Example 4.1 Case 2; Table 5 (Figure 4 uses the two-layer sin/sin solver) | `deep/circle_2d_1L.py` |
| [circle_2d_2l_sin_sin.py](../experiments/ex1_flower/ex1_case2/circle_2d_2l_sin_sin.py) | Example 4.1 Case 2; Table 5 (Figure 4 uses the two-layer sin/sin solver) | `deep/circle_2d_2L.py` |
| [circle_2d_2l_tanh_sin.py](../experiments/ex1_flower/ex1_case2/circle_2d_2l_tanh_sin.py) | Example 4.1 Case 2; Table 5 (Figure 4 uses the two-layer sin/sin solver) | `ex1_flower/ex1_case2/circle_2d_2L.py` |
| [circle_2d_3l_sin_sin_sin.py](../experiments/ex1_flower/ex1_case2/circle_2d_3l_sin_sin_sin.py) | Example 4.1 Case 2; Table 5 (Figure 4 uses the two-layer sin/sin solver) | `deep/circle_2d_3L.py` |
| [circle_2d_3l_tanh_tanh_sin.py](../experiments/ex1_flower/ex1_case2/circle_2d_3l_tanh_tanh_sin.py) | Example 4.1 Case 2; Table 5 (Figure 4 uses the two-layer sin/sin solver) | `ex1_flower/ex1_case2/circle_2d_3L.py` |
| [mix_sphere_interface.py](../experiments/ex2_sphere/mix_sphere_interface.py) | Example 4.2, mixed; Tables 14-15 | `ex2_sphere/mix_sphere_interface.py` |
| [sphere_interface_paper_final.py](../experiments/ex2_sphere/sphere_interface_paper_final.py) | Example 4.2; Tables 7-11; Figure 5 | `ex2_sphere/sphere_interface_paper_final.py` |
| [three_interface_auto_diff_paper.py](../experiments/ex3_three_interface/three_interface_auto_diff_paper.py) | Example 4.3, automatic differentiation; Table 18 | `ex3_three_interface/three_interface_auto_diff_paper.py` |
| [three_interface_paper_final.py](../experiments/ex3_three_interface/three_interface_paper_final.py) | Example 4.3; Tables 16-17,19; Figure 6 | `ex3_three_interface/three_interface_paper_final.py` |
| [square_interface_oscillate_cosine.py](../experiments/ex4_square/square_interface_oscillate_cosine.py) | Example 4.4, cosine; Tables 22-23; Figures 11-12 | `square_interface_splution_cos.py` |
| [square_interface_oscillate_solution.py](../experiments/ex4_square/square_interface_oscillate_solution.py) | Example 4.4, tanh; Tables 20-21; Figures 7-10 | `ex4_square/square_interface_oscillate_solution.py` |
| [hyper_plane_interface_paper_final.py](../experiments/ex5_high_dimension/hyper_plane_interface_paper_final.py) | Example 4.5; Tables 24-25 | `ex5_high_dimension/hyper_plane_interface_paper_final.py` |
| [heat_interface_change_paper_final.py](../experiments/ex6_moving_interface/heat_interface_change_paper_final.py) | Example 4.6; Tables 26-27; Figures 13-17 | `ex6_moving_interface/heat_interface_change_paper_final.py` |

The single-layer circle file covers tanh and sine through `--outer-activation`; its default remains the original tanh. The other four circle filenames state their actual exterior activation sequence. Source defaults and numerical expressions are retained. The Table 5 and figure configurations select the paper parameters explicitly.

## Shared Numerical Helpers

| Module | Function | Imports by retained solvers |
| --- | --- | --- |
| `finite_difference` | `fdd1` | 1 |
| `finite_difference` | `fdd2` | 1 |
| `finite_difference` | `fdux` | 9 |
| `finite_difference` | `fdux_variant2` | 5 |
| `finite_difference` | `fdux_variant3` | 1 |
| `finite_difference` | `fduxx` | 12 |
| `finite_difference` | `fduy` | 9 |
| `finite_difference` | `fduy_variant2` | 5 |
| `finite_difference` | `fduyy` | 12 |
| `finite_difference` | `fduz` | 1 |
| `finite_difference` | `fduzz` | 1 |
| `quadrature` | `gauss_points_2d` | 11 |
| `quadrature` | `gauss_points_3d` | 3 |
| `quadrature` | `gauss_weights_2d` | 11 |
| `quadrature` | `gauss_weights_3d` | 3 |
