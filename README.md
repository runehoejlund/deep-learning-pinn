# Physics-informed Neural Networks. Project in DTU Course 02456 Deep Learning. Fall 2021.
Øyvind Winton (s160602) and Rune Højlund (s173965)

Physics-informed neural networks (PINNs) combine data and mathematical models in a framework for inference and system identification. Here, we implement a PINN to solve Burger's equation and compare results to a previously implemented finite element method (FEM) solution. We find that the PINN is able to resemble the FEM solution using only the physics of the problem for training, with a mean squared error of 2.40E-4, with the main discrepancies around the shock-wave. We further demonstrate system identification on the same equation, and the flexibility of the framework by applying it to a different field with minimal modifications.

The figures of the report can be obtained by running the code in the notebooks. The main result is produced in the notebook PINN-Burgers-master.ipynb.
