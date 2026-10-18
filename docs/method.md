# Method

The paper solves elliptic interface equations with piecewise diffusion coefficients, prescribed boundary values, and jumps in solution and normal flux. The moving-interface example adds the time derivative and treats time as another network input.

Each subdomain has an independent randomized neural network. Hidden-layer parameters are initialized once from a uniform distribution and remain fixed. Their outputs supply basis functions. Finite differences approximate first, second, or directional derivatives of those basis functions. PDE residuals, boundary conditions, and interface conditions form blocks of a linear system. Penalty factors scale boundary and interface rows. NumPy or SciPy least squares computes the output-layer coefficients. Mixed variants also approximate flux components with additional randomized networks.

The finite-difference helpers deliberately retain the original arithmetic expressions, including the order of additions and multiplications. Quadrature uses tensor-product Gauss-Legendre points with the original Cartesian-product ordering. High-dimensional errors use Monte Carlo integration. Both standard and mixed sphere examples use absolute L2 errors for the paper comparisons; the other examples use relative L2 errors.

The paper's default activation is tanh. Table 5 uses one, two, or three exterior hidden layers with the documented tanh/sine sequences. The highly oscillatory square example also uses cosine activation. These choices are explicit in the corresponding solvers and configurations. See [implementation notes](implementation_notes.md) for preservation decisions.
