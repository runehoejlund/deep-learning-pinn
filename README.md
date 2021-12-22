# Physics-informed Neural Networks. Project in DTU Course 02456 Deep Learning. Fall 2021.
Øyvind Winton (s160602) and Rune Højlund (s173965)

Physics-informed neural networks (PINNs) combine data and mathematical models in a framework for inference and system identification. In this project, we implemented a PINN to solve the non-linear Burger's equation from fluid dynamics and compared the results with a finite element method (FEM) solution. We found that the PINN is able to resemble the FEM solution using only the physics of the problem for training, with a low mean squared error of 2.40 E-4. The main discrepancies was located around the shock-wave of the solution function. We further investigated how PINNs can be used for system identification on the same equation. Finally we demonstrated the flexibility of the framework by applying it to an entirely different application, the Shallow Shelf Approximation from glaciology, with only minor modifications.

## Reproduce the Main Results
The main result is produced in the notebook (PINN-Burgers-master.ipynb)[https://github.com/runehoejlund/deep-learning-pinn/blob/main/PINN-Burgers-master.ipynb]. The remaining figures of the report can be obtained by running the code in the other notebooks. 

**Running the Project**

To run the project you need a python installation (e.g. the Anaconda distribution) and an IDE for opening Jupyter Notebooks. You can install all requirements by running the following command in terminal from within the project directory:
```
pip install -r requirements.txt
```