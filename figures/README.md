# Figures

Run `python scripts/run_experiment.py configs/figure_03.json` for Figure 3, or select another figure configuration. Figures 13-17 share `configs/figures_13_17.json` and one space-time solve.

The runner saves `figures/<name>/figure_XX.png`, `figure_XX.pdf`, and `figure_XX.npz`. The archive contains the numerical solution, exact solution, and absolute-error grid. The three-panel exports use the original rainbow colormap and coordinate orientation. Layout and file metadata are newly generated; they are not copies of the publisher's images.

Legacy plots emitted by `plt.show()` are saved separately as `run_XX_panel_YY.png`. Several original scripts return before their commented plotting code, which is why the new postprocessing module evaluates the fitted models separately. It uses one-point network evaluation, preserving the original evaluation matrix sizes.

Figure 5 is the z=0 spherical cross-section. Figures 13-17 use the moving interface radius `0.5 + 0.3*t` at t=0, 0.25, 0.5, 0.75, 1. The original commented final-time plotting block used the t=0.75 mask for t=1; that inactive block remains unchanged. The new exporter uses the interface equation from the paper.

The smoke figure uses a smaller grid and is not a paper result. All generated figure directories are ignored by Git.
