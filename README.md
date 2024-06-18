# chicken_swarm_quantum

# IN PROGRESS!!!!

A 'quantum' particle swarm optimizer written in Python using quantum inspired methods (non-qiskit version).  Modified from the [adaptive timestep PSO optimizer](https://github.com/jonathan46000/pso_python) by [jonathan46000](https://github.com/jonathan46000) to keep a consistent format across optimizers in AntennaCAT.


Now featuring AntennaCAT hooks for GUI integration and user input handling.
 

This branch of the repository is part of a series of replication studies. It is not intended to be the 'best' implementation, or the most recent update. This method has been chosen to replicate a specific snapshot of literature.


## Table of Contents
* [Chicken Swarm Optimization](#chicken-swarm-optimization)
* [Quantum Inspired Optimization](#quantum-inspired-optimization)
* [Quantum Particle Swarm Optimization](#quantum-particle-swarm-optimization)
    * [Mean Best Position](#mean-best-position)
    * [Position Update](#position-update)
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Constraint Handling](#constraint-handling)
    * [Boundary Types](#boundary-types)
    * [Multi-Object Optimization](#multi-object-optimization)
    * [Objective Function Handling](#objective-function-handling)
      * [Internal Objective Function Example](internal-objective-function-example)
* [Examples](#example-implementations)
    * [Basic PSO Example](#basic-pso-example)
    * [Detailed Messages](#detailed-messages)
    * [Realtime Graph](#realtime-graph)
* [References](#references)
* [Publications and Integration](#publications-and-integration)
* [Licensing](#licensing)  

## Chicken Swarm Optimization

The Chicken Swarm Optimization (CSO) algorithm, introduced by Meng et al. in 2014 [1], is inspired by the hierarchy and behaviors observed in a swarm of chickens, including roosters, hens, and chicks. Each type of bird has its own unique movement rules and interactions.

In CSO, there is an absence of a direct random velocity component, which is an important distinction from many PSO variations, though it is not the only variation without a velocity vector. Instead, the movements are based on specific behaviors and social interactions within the swarm. Generally, the movement rules for each type of bird in CSO can be described as:

1) Roosters:

    Roosters have the best positions (fitness values) in the swarm. They move based on their current position and the random perturbation to avoid getting stuck in local optima.

2) Hens:

    Hens follow roosters. They update their positions based on the positions of the roosters they follow and a randomly selected chicken. This reflects the social hierarchy and interaction in the swarm.

3) Chicks:

    Chicks follow their mother hens. They update their positions based on their mother's positions with some random factor to simulate the dependent behavior.

## Quantum Inspired Optimization

Quantum Particle Swarm Optimization (QPSO) was introduced in 2004 [2] [3]. Paraphrased from [2], in PSO the location and velocity vectors are used to determine the trajectory of the particle, which because in Newtonian mechanics a particle moves along a determined trajectory. However, in quantum mechanics, the location and velocity vectors cannot be determined/known simultaneously due to uncertainty principle (Werner Heisenberg, 1927). The takeaway being, quantum-inspired algorithms take concepts from quantum mechanics, such as superposition and entanglement, and apply them in classical computation to solve optimization problems more effectively (depending on the problem type). These two concepts are applied, generally speaking, as follows:

1) Superposition

**Quantum Concept**: In quantum mechanics, a particle can exist in a superposition of multiple states simultaneously. For example, a quantum bit (qubit) can be in a state ∣0⟩∣0⟩, ∣1⟩∣1⟩, or any linear combination ∣ψ⟩=α∣0⟩+β∣1⟩∣ψ⟩=α∣0⟩+β∣1⟩, where α and β are complex numbers.

**Classical Adaptation**: In quantum-inspired algorithms, superposition can be interpreted as a probability distribution over multiple states. Instead of particles having a single position, they are represented by a probability distribution, reflecting the potential to be in various positions simultaneously.

**Example in QPSO**: In the Quantum-inspired Particle Swarm Optimization, a particle’s position is often updated using a probability distribution derived from both personal best and global best positions, rather than a deterministic position update. This allows particles to explore the search space more effectively.

2) Entanglement

**Quantum Concept**: Entanglement is a phenomenon where particles become interconnected such that the state of one particle directly affects the state of another, no matter the distance between them. This creates a strong correlation between the particles.

**Classical Adaptation**: In quantum-inspired algorithms, entanglement can be represented as a dependency or correlation between particles. When one particle updates its position, it influences the position updates of other particles, promoting cooperative behavior among the particles in the swarm.

**Example in QPSO**: In QPSO, the positions of particles might be updated using a combination of their personal best position and the global best position, creating a form of "entanglement" where particles are influenced by the best solutions found by the swarm, thus maintaining a level of coordination and cooperation.


## Quantum Chicken Swarm Optimization

Unlike traditional PSO, Quantum Particle Swarm Optimization (QPSO) doesn't use a velocity vector. Instead, it updates particle positions directly based on a probability distribution based on the mean best position and a logarithmic factor, which has roots in the quantum mechanics principles mentioned previously. The QPSO update rule leverages quantum-inspired probabilistic movements to balance exploration and exploitation. By combining the best aspects of personal and global experiences and adding a stochastic component, QPSO can effectively search complex optimization landscapes. 

For Quantum Chicken Swarm Optimization (QCSO), the movement rules are respected from the classical versions, but the position changes are implemented slightly different. In [4]   


Generally, the movement rules for each type of bird in QCSO can be described as:

### Quantum Roosters

    Roosters have the best positions (fitness values) in the swarm. They move based on their current position and the random perturbation to avoid getting stuck in local optima.

    This implementation has a toggle for using classical or quantum behavior for the roosters.

    The quantum-inspired implementation ....







### Quantum Hens

    Hens follow roosters. They update their positions based on the positions of the roosters they follow and a randomly selected chicken. This reflects the social hierarchy and interaction in the swarm.

    The quantum-inspired implementation ....





### Quantum Chicks

    Chicks follow their mother hens. They update their positions based on their mother's positions with some random factor to simulate the dependent behavior.


    The quantum-inspired implementation ....












## Requirements

This project requires numpy and matplotlib. 

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.2.1
cycler==0.12.1
fonttools==4.51.0
importlib_resources==6.4.0
kiwisolver==1.4.5
matplotlib==3.8.4
numpy==1.26.4
packaging==24.0
pillow==10.3.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
six==1.16.0
zipp==3.18.1

```

## Implementation
### Constraint Handling
Users must create their own constraint function for their problems, if there are constraints beyond the problem bounds.  This is then passed into the constructor. If the default constraint function is used, it always returns true (which means there are no constraints).

### Boundary Types
This optimizers has 4 different types of bounds, Random (Particles that leave the area respawn), Reflection (Particles that hit the bounds reflect), Absorb (Particles that hit the bounds lose velocity in that direction), Invisible (Out of bound particles are no longer evaluated).

Some updates have not incorporated appropriate handling for all boundary conditions.  This bug is known and is being worked on.  The most consistent boundary type at the moment is Random.  If constraints are violated, but bounds are not, currently random bound rules are used to deal with this problem. 

### Multi-Object Optimization
The no preference method of multi-objective optimization, but a Pareto Front is not calculated. Instead the best choice (smallest norm of output vectors) is listed as the output.

### Objective Function Handling
The optimizer minimizes the absolute value of the difference from the target outputs and the evaluated outputs.  Future versions may include options for function minimization absent target values. 

#### Internal Objective Function Example
The current internal optimization function takes 3 inputs, and has 2 outputs. It was created as a simple 3-variable optimization objective function that would be quick to converge.  
<p align="center">
        <img src="https://github.com/LC-Linkous/cat_swarm_python/blob/main/media/obj_func_pareto.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 

This function has three files:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.

Other multi-objective functions can be applied to this project by following the same format (and several have been collected into a compatible library, and will be released in a separate repo)


## Example Implementations

### Basic Swarm Example
main_test.py provides a sample use case of the optimizer. 

### Detailed Messages
main_test_details.py provides an example using a parent class, and the self.suppress_output and detailedWarnings flags to control error messages that are passed back to the parent class to be printed with a timestamp. This implementation sets up the hooks for integration with AntennaCAT in order to provide the user feedback of warnings and errors.

### Realtime Graph

<p align="center">
        <img src="https://github.com/LC-Linkous/chicken_swarm_python/blob/main/media/chicken_swarm.gif" alt="Example Chicken Swarm Optimizer" height="200">
</p>

main_test_graph.py provides an example using a parent class, and the self.suppress_output and detailedWarnings flags to control error messages that are passed back to the parent class to be printed with a timestamp. Additionally, a realtime graph shows particle locations at every step. 

NOTE: if you close the graph as the code is running, the code will continue to run, but the graph will not re-open.

## References

[1] X. B. Meng, Y. Liu, X. Gao, and H. Zhang, "A new bio-inspired algorithm: Chicken swarm optimization," in Proc. Int. Conf. Swarm Intell. Cham, Switzerland, Springer, 2014, pp. 86–94.

[2] Jun Sun, Bin Feng and Wenbo Xu, "Particle swarm optimization with particles having quantum behavior," Proceedings of the 2004 Congress on Evolutionary Computation (IEEE Cat. No.04TH8753), Portland, OR, USA, 2004, pp. 325-331 Vol.1, doi: 10.1109/CEC.2004.1330875.

[3] Jun Sun, Wenbo Xu and Bin Feng, "A global search strategy of quantum-behaved particle swarm optimization," IEEE Conference on Cybernetics and Intelligent Systems, 2004., Singapore, 2004, pp. 111-116 vol.1, doi: 10.1109/ICCIS.2004.1460396.
  
[4] Z. Qiuqiao, B. Wang, L. Wei and W. Haishan, "Chicken swarm optimization algorithm based on quantum behavior and its convergence analysis," 2020 39th Chinese Control Conference (CCC), Shenyang, China, 2020, pp. 2107-2112, doi: 10.23919/CCC50068.2020.9189572.

[5] IEEE link


## Publications and Integration
This software works as a stand-alone implementation, and as one of the optimizers integrated into AntennaCAT.

Publications featuring the code in this repo will be added as they become public.

## Licensing

The code in this repository has been released under GPL-2.0


