#! /usr/bin/python3

##--------------------------------------------------------------------\
#   chicken_swarm_python
#   '.src/lundquist_3_var/constr_F.py'
#   Function for objective function constraints.
#   Has 2 checks: 1 for the function limitations, 1 for float size
#   Returns True if x array passes constraints check, False otherwise   
#
#   Author(s): Jonathan Lundquist, Lauren Linkous
#   Last update: May 28, 2024
##--------------------------------------------------------------------\


def constr_F(X):
    F = True
    # objective function/problem constraints
    if (X[2] > X[0]/2) or (X[2] < 0.1):
        F = False

    return F