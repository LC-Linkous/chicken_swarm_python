#! /usr/bin/python3

##--------------------------------------------------------------------\
#   chicken_swarm_quantum
#   './chicken_swarm_quantum/src/main_test.py'
#   Test function/example for using the 'swarm' class in cat_swarm.py.
#       This has been modified from the original to include message 
#       passing back to the parent class or testbench, rather than printing
#       error messages directly from the 'swarm' class. Format updates are 
#       for integration in the AntennaCAT GUI.
#
#   Author(s): Lauren Linkous
#   Last update: June 28, 2025
##--------------------------------------------------------------------\


import pandas as pd
import numpy as np
from chicken_swarm import swarm

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs     # multi objective function


if __name__ == "__main__":
    # constant variables
    TOL = 10 ** -6                      # Convergence Tolerance
    MAXIT = 10000                       # Maximum allowed iterations
    BOUNDARY = 1                        # int boundary 1 = random,      2 = reflecting
                                        #              3 = absorbing,   4 = invisible



    # Objective function dependent variables
    LB = func_configs.LB                    # Lower boundaries, [[0.21, 0, 0.1]]
    UB = func_configs.UB                    # Upper boundaries, [[1, 1, 0.5]]
    IN_VARS = func_configs.IN_VARS          # Number of input variables (x-values)   
    OUT_VARS = func_configs.OUT_VARS        # Number of output variables (y-values)
    TARGETS = func_configs.TARGETS          # Target values for output
    # target format. TARGETS = [0, ...] 

    # threshold is same dims as TARGETS
    # 0 = use target value as actual target. value should EQUAL target
    # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
    # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
    #DEFAULT THRESHOLD
    THRESHOLD = np.zeros_like(TARGETS) 
    #THRESHOLD = np.ones_like(TARGETS)
    #THRESHOLD = [0, 1, 0]


    # Objective function dependent variables
    func_F = func_configs.OBJECTIVE_FUNC  # objective function
    constr_F = func_configs.CONSTR_FUNC   # constraint function

    
    # chicken swarm specific
    RN = 10                       # Total number of roosters
    HN = 20                       # Total number of hens
    MN = 15                       # Number of mother hens in total hens
    CN = 20                       # Total number of chicks
    G = 70                        # Reorganize groups every G steps 

    # quantum swarm variables
    BETA = 0.5                  #Float constant controlling influence 
                                    #between the personal and global best positions
    QUANTUM_ROOSTERS = True     # Boolean. Use quantum rooster or classical movement


    best_eval = 1
    parent = None             # for the optimizer test ONLY
    evaluate_threshold = True # use target or threshold. True = THRESHOLD, False = EXACT TARGET
    suppress_output = True    # Suppress the console output of particle swarm
    allow_update = True       # Allow objective call to update state 

    # Constant variables
    opt_params = {'BOUNDARY': [BOUNDARY],                   # int boundary 1 = random,      2 = reflecting
                                                            #              3 = absorbing,   4 = invisible
                'RN': [RN],                                 # Total number of roosters
                'HN': [HN],                                 # Total number of hens
                'MN': [MN],                                 # Number of mother hens in total hens
                'CN': [CN],                                 # Total number of chicks
                'G': [G],                                   # Reorganize groups every G steps 
                'BETA': [BETA],                             # Float constant controlling influence 
                                                            #     between the personal and global best positions 
                'QUANTUM_ROOSTERS': [QUANTUM_ROOSTERS]}     # Boolean. Use quantum rooster or classical movement
    
    opt_df = pd.DataFrame(opt_params)
    mySwarm = swarm(LB, UB, TARGETS, TOL, MAXIT,
                            func_F, constr_F,
                            opt_df,
                            parent=parent, 
                            evaluate_threshold=evaluate_threshold, obj_threshold=THRESHOLD)  
 
    

    # instantiation of particle swarm optimizer 
    while not mySwarm.complete():

        # step through optimizer processing
        # update_velocity, will change the particle location
        mySwarm.step(suppress_output)

        # call the objective function, control 
        # when it is allowed to update and return 
        # control to optimizer

        # for some objective functions, the function
        # might not evaluate correctly (e.g., under/overflow)
        # so when that happens, the function is not evaluated
        # and the 'step' fucntion will re-gen values and try again

        mySwarm.call_objective(allow_update)
        iter, eval = mySwarm.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        if suppress_output:
            if iter%100 ==0: #print out every 100th iteration update
                print("Iteration")
                print(iter)
                print("Best Eval")
                print(best_eval)

    print("Optimized Solution")
    print(mySwarm.get_optimized_soln())
    print("Optimized Outputs")
    print(mySwarm.get_optimized_outs())

