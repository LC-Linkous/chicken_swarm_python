#! /usr/bin/python3

##--------------------------------------------------------------------\
#   2022_improved_chicken_swarm_python
#   './chicken_swarm_python/src/chicken_swarm.py'
#   2022 'improved' chicken swarm optimization (ICSO) class.
#   Based on:
#       J. Liang, L. Wang, and M. Ma, "An Improved Chicken Swarm
#       Optimization Algorithm for Solving Multimodal Optimization
#       Problems," Computational Intelligence and Neuroscience,
#       vol. 2022, Article ID 5359732, 2022.
#       https://doi.org/10.1155/2022/5359732
#
#   The ICSO adds two mechanisms to the standard (Meng et al. 2014) CSO:
#   1) RECSO (paper Sec. 3.1): at each role update (after the first), the
#      BFA-inspired reproduction operation replaces the chicks (per the
#      PREVIOUS role assignment) with copies of the same number of
#      best-performing individuals (Eq. 7), then the elimination-dispersal
#      operation scatters those chicks to random positions in the search
#      space with probability Ped = 0.25 (Eq. 8). Roles are then
#      re-assigned by fitness.
#   2) CSO-PSO (paper Sec. 3.2/3.3): at EVERY iteration (one full cycle
#      through the population in this state-machine framework), the swarm
#      is randomly re-divided into two equal-scale subgroups. Subgroup 1
#      moves by the (RE)CSO role rules; subgroup 2 moves by standard PSO
#      velocity-position updates (c1 = c2 = 2, paper Sec. 4.1.2). The
#      subgroups share one population, one fitness record, and one global
#      best, which implements the paper's merge/information-exchange step.
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 11, 2026
##--------------------------------------------------------------------\


import numpy as np
from numpy.random import Generator, MT19937
import sys
np.seterr(all='raise')


class swarm:
    # arguments should take the form: 
    # swarm([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj, 
    # bool, [int, int, ...], 
    # int) 
    #  
    # opt_df contains class-specific tuning parameters
    # boundary: int. 1 = random, 2 = reflecting, 3 = absorbing,   4 = invisible
    # RN: int
    # HN: int
    # MN: int
    # CN: int
    # G: int
    # w_min: float (PSO inertia weight, minimum)
    # w_max: float (PSO inertia weight, maximum/starting)
    # c1: float (PSO cognitive learning factor, personal best)
    # c2: float (PSO social learning factor, global best)
    # ped: float (elimination-dispersal probability, 0-1)
    #

    def __init__(self,  lbound, ubound, targets, E_TOL, maxit,
                 obj_func, constr_func, 
                 opt_df,
                 parent=None, 
                 evaluate_threshold=False, obj_threshold=None,
                 decimal_limit = 4): 
        

        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent 


        self.number_decimals = int(decimal_limit)  # limit the number of decimals
                                              # used in cases where real life has limitations on resolution



        #evaluation method for targets
        # True: Evaluate as true targets
        # False: Evaluate as thesholds based on information in obj_threshold
        if evaluate_threshold==False:
            self.evaluate_threshold = False
            self.obj_threshold = None

        else:
            if not(len(obj_threshold) == len(targets)):
                self.debug_message_printout("WARNING: THRESHOLD option selected.  +\
                Dimensions for THRESHOLD do not match TARGET array. Defaulting to TARGET search.")
                self.evaluate_threshold = False
                self.obj_threshold = None
            else:
                self.evaluate_threshold = evaluate_threshold #bool
                self.obj_threshold = np.array(obj_threshold).reshape(-1, 1) #np.array
        


        #unpack the opt_df standardized vals
        boundary = int(opt_df['BOUNDARY'][0])
        RN = int(opt_df['RN'][0])
        HN = int(opt_df['HN'][0])
        MN = int(opt_df['MN'][0])
        CN = int(opt_df['CN'][0])
        G = int(opt_df['G'][0])
        NO_OF_PARTICLES = RN + HN + MN + CN
        W_min = float(opt_df['MIN_WEIGHT'][0])
        W_max = float(opt_df['MAX_WEIGHT'][0])
        C1 = float(opt_df['C1'][0])     # PSO cognitive learning factor
        C2 = float(opt_df['C2'][0])     # PSO social learning factor
        PED = float(opt_df['PED'][0])   # elimination-dispersal probability



        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        lbound = np.array(lbound[0])
        ubound = np.array(ubound[0])

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):
            
            if self.parent == None:
                pass
            else:
                self.parent.record_params()
                self.parent.debug_message_printout("ERROR: lbound and ubound must be 1xN-dimensional \
                                                        arrays  with the same length")
           
        else:
        
            if heightl == 1:
                lbound = lbound
        
            if heightu == 1:
                ubound = ubound

            self.lbound = lbound
            self.ubound = ubound
            variation = ubound-lbound


            # chicken swarm vars
            # population split
            self.RN = RN    # rooster number
            self.HN = HN    # hen number
            self.MN = MN    # mother hen number
            self.CN = CN    # chick number
            total_chickens = RN + HN + MN + CN

            self.G = G                       # num full cycles before updating the 
            self.G_steps = G*total_chickens  # how many iterations happen in a generation
 
            # error checking on population split
            if (NO_OF_PARTICLES < total_chickens):
                self.parent.debug_message_printout("WARNING: number of chickens adds up to more \
                                                   than the expected number of particles. Attempting fix.")
                # subtract number of roosters and chicks. rest are hens
                num_hens = NO_OF_PARTICLES - RN - CN - MN
                if (num_hens <=0):
                    self.parent.debug_message_printout("ERROR: total of roosters and chicks is higher than \
                                                       the expected swarm size. Swarm init failed.")
                    return
                else:
                    if (num_hens > MN):
                        self.MN = MN
                        self.HN = num_hens - MN # extra non-mother hens
                        self.parent.debug_message_printout("WARNING: reduced number of hens used for swarm initiation.\
                                                            Number of mother hens is " + str(self.MN) +\
                                                            ". Total number of hens is " + str(num_hens))
            elif (NO_OF_PARTICLES > total_chickens):
                    num_hens = NO_OF_PARTICLES - total_chickens
                    self.HN = self.HN + num_hens # extra non-mother hens
                    self.parent.debug_message_printout("WARNING: there are fewer than expected chickens, adding extra hens.\
                                    Check categories for chickens.")

            elif (MN < 1) and (CN>0): #chicks but no mother hens
                self.parent.debug_message_printout("ERROR: there are chicks, but no mother hens.\
                Check categories for chickens.")
                return
   
            #split chickens into groups           
            # Each group has a rooster (chicken with best fitness value)
            # The worst preforming agents are chicks.
            # The rest are hens. hens randomly chose a group to liven in.
            # mother-child relation in randomly established
            
            # assign which group the chickens are in 
            # first chicken is a rooster, in case there's only 1 searching agent
            # chicken_info = array of [CLASSIFICATION(0-4), GROUP(0-(RN-1)), MOTHER-HEN-ID]
            # classificition: 0 = rooster, 1 = hen, 2 = mother hen, 3 = chicks
            self.chicken_info = np.array([0, 0, -1])

            #randomly initialize the positions
            # NOTE: the first particle is initialized with the SAME form as the loop
            # below. The previous version used rng.random((max(h,w),1)) here, which
            # broadcast to a malformed (N,N) first row and collapsed self.M's shape.
            self.M = np.round(np.array(np.multiply(self.rng.random((1,np.max([heightl, widthl]))), variation)+lbound), self.number_decimals)


            if NO_OF_PARTICLES > 1:
                # make temp classification list
                classList = np.zeros(NO_OF_PARTICLES)
                # rooster
                classList[0:RN] = 0
                # hen
                start = RN
                end = RN + HN
                classList[start:end] = 1
                # mother hen
                start = RN + HN
                end = RN + HN + MN
                classList[start:end] = 2
                # chicks
                start = RN + HN + MN
                classList[start:] = 3
                # array of groups to use in random allocation
                group_nums = np.arange(RN)

            for i in range(2,int(NO_OF_PARTICLES)+1):
                # set initial location
                self.M = \
                    np.round(np.vstack([self.M, 
                               np.multiply( self.rng.random((1,np.max([heightl, widthl]))), 
                                                                               variation) 
                                                                               + lbound]), self.number_decimals)
                if classList[i-1] == 0: #rooster
                    # assign to the next group (i-1), and done.
                    self.chicken_info = \
                        np.vstack([self.chicken_info, 
                                [classList[i-1], i-1, -1]])  #[ChickenTypeID, groupNumber, childNumTag]

                elif (classList[i-1] == 1) or (classList[i-1] == 2): #hen, mother hen
                    # assign to a random group.
                    hen_group = self.rng.choice(group_nums)
                    self.chicken_info = \
                        np.vstack([self.chicken_info, 
                                [classList[i-1], hen_group, -1]])

                elif classList[i-1] == 3: #chick
                    # select a random hen to be the 'mother' and assign to group
                    groupAssigned = False

                    while (groupAssigned == False):
                        chicken_idx = self.rng.integers(0, i-1)# index after a chick will always be the chick
                        if self.chicken_info[chicken_idx][0] == 2: #is mother hen
                            # get the group the random mother hen is from
                            mother_group = self.chicken_info[chicken_idx][1]         
                            # assign chick to group, and to that mother hen                    
                            self.chicken_info = \
                                np.vstack([self.chicken_info, 
                                        [classList[i-1], mother_group, chicken_idx]])
                            groupAssigned = True


            '''
            self.M                      : An array of current particle (cat) locations.
            self.RN                     : Number of roosters. Integer.
            self.HN                     : Total number of hens. Integer. 
            self.MN                     : Number of mother hens. Integer.
            self.CN                     : Number of chicks. Integer.  
            self.chicken_info           : classification (R,H,C), group #, mother ID. Array.          
            self.G                      : How often to randomize groups. Integer. Num full cycles of chickens
            self.output_size            : An integer value for the output size of obj func
            self.Active                 : An array indicating the activity status of each particle. (e.g., in bounds)
            self.Gb                     : Global best position, initialized with a large value.
            self.F_Gb                   : Fitness value corresponding to the global best position.
            self.Pb                     : Personal best position for each particle.
            self.F_Pb                   : Fitness value corresponding to the personal best position for each particle.
            self.targets                : Target values for the optimization process.
            self.maxit                  : Maximum number of iterations.
            self.E_TOL                  : Error tolerance.
            self.obj_func               : Objective function to be optimized.      
            self.constr_func            : Constraint function.  
            self.iter                   : Current iteration count.
            self.current_particle       : Index of the current particle being evaluated.
            self.number_of_particles    : Total number of particles. 
            self.allow_update           : Flag indicating whether to allow updates.
            self.boundary               : Boundary conditions for the optimization problem.
            self.Flist                  : List to store fitness values.
            self.Fvals                  : List to store fitness values.
            self.Mlast                  : Last location of particle
            self.W_max                  : Constant float. maximum, starting PSO inertia weight
            self.W_min                  : Constant float. minimum PSO inertia weight.
            self.W                      : (inertia) Weight of current particle. Decreases W_max -> W_min
            self.C1                     : PSO cognitive learning factor. Weight of personal best (Pb) term.
            self.C2                     : PSO social learning factor. Weight of global best (Gb) term.
            self.PED                    : Probability of elimination-dispersal for replicated chicks. 0-1.
            self.V                      : PSO velocity array. Same shape as self.M.
            self.pso_flags              : Bool array. True = particle is in the PSO subgroup this cycle.
            self.ran_reorganize         : Bool. Whether a fitness-based role assignment has happened.
            '''

            self.output_size = len(targets)
            self.Active = np.ones((NO_OF_PARTICLES))                        
            self.Gb = sys.maxsize*np.ones((1,np.max([heightl, widthl])))   
            self.F_Gb = sys.maxsize*np.ones((1,self.output_size))                
            self.Pb = sys.maxsize*np.ones(np.shape(self.M))                 
            self.F_Pb = sys.maxsize*np.ones((NO_OF_PARTICLES,self.output_size))  
            self.targets = np.array(targets).reshape(-1, 1)                    
            self.maxit = maxit                                             
            self.E_TOL = E_TOL                                              
            self.obj_func = obj_func                                             
            self.constr_func = constr_func                                   
            self.iter = 0                                                   
            self.current_particle = 0                                       
            self.number_of_particles = NO_OF_PARTICLES                      
            self.allow_update = 0                                           
            self.boundary = boundary                                       
            self.Flist = []                                                 
            self.Fvals = []                                                 
            self.Mlast = 1*self.ubound            
            self.W_max = W_max
            self.W_min = W_min              
            self.W = W_max  # linearly decreases to W_min. see pso_update()
            self.C1 = C1            
            self.C2 = C2
            self.PED = PED
            # PSO velocity array. initialized to zeros so that the first 
            # PSO contribution is driven by the Pb/Gb terms only
            self.V = np.zeros(np.shape(self.M))
            # CSO/PSO subgroup membership flags. re-randomized at the start
            # of every full cycle through the population. True = the
            # particle moves by the PSO update this cycle (subgroup 2),
            # False = the particle moves by the CSO role rules (subgroup 1)
            self.pso_flags = np.zeros((NO_OF_PARTICLES), dtype=bool)
            # safety flag. set when every particle has gone inactive
            # (possible with INVISIBLE boundaries) so complete() can end
            # the run instead of letting the driver loop forever
            self.swarm_stalled = False
            # tracks whether a fitness-based role assignment has happened
            # yet. the reproduction & elimination-dispersal operations are
            # skipped at the FIRST role update (paper Sec. 3.3 step 5(a)),
            # when fitness-based roles have not been assigned
            self.ran_reorganize = False
                                         

            self.debug_message_printout("swarm successfully initialized")
            

    def call_objective(self, allow_update):
        if self.Active[self.current_particle]:
            # call the objective function. If there's an issue with the function execution, 'noError' returns False
            newFVals, noError = self.obj_func(self.M[self.current_particle], self.output_size)
            if noError == True:
                self.Fvals = np.array(newFVals).reshape(-1, 1)
                if allow_update:
                    # EVALUATE OBJECTIVE FUNCTION - TARGET OR THRESHOLD
                    self.Flist = self.objective_function_evaluation(self.Fvals, self.targets)# abs(self.targets - self.Fvals)
                    self.iter = self.iter + 1
                    self.allow_update = 1
                else:
                    self.allow_update = 0
            return noError# return is for error reporting purposes only


    def objective_function_evaluation(self, Fvals, targets):
        #pass in the Fvals & targets so that it's easier to track bugs

        # this uses the fitness values and target (or threshold) to determine the Flist values
        # Option #1: TARGET
        # get DISTANCE FROM TARGET
        # Option #2: THRESHOLD
        # use THRESHOLD TO DETERMINE INTEREST
        # if threshold is met, the distance is set to a small value (epsilon).
        #  Setting the 'distance' to epsilon, the convergence value check can
        # also remain the same format. 


        # testing different values of epsilon
        epsilon = np.finfo(float).eps #smallest system constant
        # Ex: 2.220446049250313e-16  
        # #may be greater than tolerance if tolerance is set very low for testing
        #epsilon = 10**-18
        #epsilon = 0  # causes issues with imag. numbers

        Flist = np.zeros(len(Fvals))


        if self.evaluate_threshold == True: #THRESHOLD
            ctr = 0
            for i in targets:
                o_thres = int(self.obj_threshold[ctr].item()) #force type as err check (NumPy 2 safe)
                t = targets[ctr].item()
                fv = Fvals[ctr].item()

                if o_thres == 0: #TARGET. default
                    # sets Flist[ctr] as abs distance of  Fvals[ctr] from target
                    Flist[ctr] = abs(t - fv)

                elif o_thres == 1: #LESS THAN OR EQUAL 
                    # checks if the Fvals[ctr] is LESS THAN OR EQUAL to target
                    # if yes, then distance is 0 (considered 'on target)
                    # if no, then Flist is abs distance of  Fvals[ctr] from target
                    if fv <= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)

                elif o_thres == 2: #GREATER THAN OR EQUAL
                    # checks if the Fvals[ctr] is GREATER THAN OR EQUAL to target
                    # if yes, then distance is 0 (considered 'on target)
                    # if no, then Flist is abs distance of  Fvals[ctr] from target
                    if fv >= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)

                else: #o_thres == 0. #TARGET. default
                    self.parent.debug_message_printout("ERROR: unrecognized threshold value. Evaluating as TARGET")
                    Flist[ctr] = abs(t - fv)

                ctr = ctr + 1

        else: #TARGET as default
            # arrays are already the same dimensions. 
            # no need to loop and compare to anything
            Flist = abs(targets - Fvals)

        return Flist


    # MOVEMENT MODELS

    def move_rooster(self, particle):
        # epsilon = 'smallest system constant'. improvised.
        epsilon = 10e-50 

        # choose a random rooster
        rooster_arr = np.arange(self.RN)
        random_rooster_idx = self.rng.choice(rooster_arr)
        # use L2 norm for fitness to account for multi-objective funcs
        random_rooster_fitness = np.linalg.norm(self.F_Pb[random_rooster_idx])
        
        this_rooster_fitness = np.linalg.norm(self.F_Pb[particle])

        if this_rooster_fitness <= random_rooster_fitness:
            sig_squared = 1
        else:
            # exp((fitness_random_rooster - fitness_this_rooster)/(abs(fitness_this_rooster)-epsilon))
            #sig_squared = np.exp((random_rooster_fitness-this_rooster_fitness)/(abs(this_rooster_fitness)+epsilon))
            # -709.00 and 709.00 are the integer limits to np.exp() on system that handles float64 at most (Windows)
            clipped_val = np.clip(((random_rooster_fitness-this_rooster_fitness)/(abs(this_rooster_fitness)+epsilon)), -709.00, 709.00)
            sig_squared = np.exp(clipped_val)


        #update new location based on random()
        self.M[particle] = np.round(self.M[particle]*(1+self.rng.normal(0, sig_squared)), self.number_decimals)
    

    def move_hen(self, particle):
        #newLoc = oldLoc 
        # + S1*RANDOM(0-to-1)*(LocationRoosterGroupmate-thisChickenLocation)
        #  + S2*RANDOM(0-to-1)*(LoctionRandomChickenInSwarm-thisChickenLocation)
        # where:
        #   S1 = exp((FitnessThisChicken-FitnessRoosterGroupmate)/(abs(FitnessThisChicken)+epsilon))
        #   S2 = exp(FitnessRandomChickenInSwarm-FitnessThisChicken)
        # NOTE: FitnessRoosterGroupmate and FitnessRandomChickenInSwarm cannot be the same chicken

        # get the rooster information
        group_rooster_idx = int(self.chicken_info[particle][1]) #also the group index
        rooster_loc = self.M[group_rooster_idx]
        fitness_rooster = np.linalg.norm(self.F_Pb[group_rooster_idx])
        
        # get the random chicken information
        # initial random
        random_chicken_idx = self.rng.integers(0, self.number_of_particles)
        # random cannot be the idx of the rooster, the current chicken, or be from a chick
        while (random_chicken_idx == group_rooster_idx) or \
                        (random_chicken_idx == particle) or \
                        (int(self.chicken_info[random_chicken_idx][0]) == 3):
            random_chicken_idx = self.rng.integers(0, self.number_of_particles)

        random_chicken_loc = self.M[random_chicken_idx]
        fitness_random_chicken = np.linalg.norm(self.F_Pb[random_chicken_idx])

        fitness_this_chicken = np.linalg.norm(self.F_Pb[particle])

        # epsilon = 'smallest system constant'. improvised.
        epsilon = 10e-30 

        # exp((FitnessThisChicken-FitnessRoosterGroupmate)/(abs(FitnessThisChicken)+epsilon))
        #S1 = np.exp((fitness_this_chicken-fitness_rooster)/(np.abs(fitness_this_chicken) + epsilon))
        clipped_val = np.clip(((fitness_this_chicken-fitness_rooster)/(np.abs(fitness_this_chicken) + epsilon)), -700.00, 700.00)
        S1 = np.exp(clipped_val)
        # S1*RANDOM(0-to-1)*(LocationRoosterGroupmate-thisChickenLocation)
        # these terms can overflow because of the exp()
        # some S2 results are evaluating on the scale of 8.218407461554972e+307
        # these clipped bounds are effectively are zero and inf
        # term_1 = S1*self.rng.uniform(0,1)*(rooster_loc-self.M[particle])
        # still dealing with overflow issues. apply cap to S1
        if S1 > 10e30:
            S1 = 10e30
        elif S1 < -10e30:
            S1 = -10e30
        clipped_term1 = np.clip((S1*self.rng.uniform(0,1)*(rooster_loc-self.M[particle])), -10e30, 10e10)
        term_1 = clipped_term1

        #S2 = np.exp(float(fitness_random_chicken-fitness_this_chicken))
        #np.exp(...) throws overflow errors. Using clip as a generic catch
        clipped_val = np.clip((fitness_random_chicken-fitness_this_chicken), -700.00, 700.00)
        S2 = np.exp(clipped_val)
        #S2*RANDOM(0-to-1)*(LoctionRandomChickenInSwarm-thisChickenLocation)

        # these terms can overflow because of the exp()
        # some S2 results are evaluating on the scale of 8.218407461554972e+307
        # these clipped bounds are effectively are zero and inf
        #term_2 = S2*self.rng.uniform(0,1)*(random_chicken_loc-self.M[particle])
        # This still causes overflow:
        # clipped_term2 = np.clip((S2*self.rng.uniform(0,1)*(random_chicken_loc-self.M[particle])), -10e50, 10e10)
        if S2 > 10e30:
            S2 = 10e30
        elif S2 < -10e30:
            S2 = -10e30

        clipped_term2 = np.clip((S2*self.rng.uniform(0,1)*(random_chicken_loc-self.M[particle])), -10e30, 10e10)
        term_2 = clipped_term2

        # new_loc = old_loc + term_1 + term_2
        self.M[particle] = np.round(self.M[particle] + term_1 + term_2, self.number_decimals)


    def move_chick(self, particle):

        # The 2022 ICSO retains the STANDARD chick update. (paper Eq. 6)
        # The depth-search improvement for chicks in this variant comes from
        # the reproduction & elimination-dispersal operations at role update
        # (see replication_elimination_dispersal()), NOT from a modified
        # movement equation as in the 2015 improved chicken swarm.
        # nextLoc = currentLoc + FL*(locationMother - currentLoc)
        # NOTE: per paper Eq. 6, FL is a following coefficient drawn from
        # (0,2). Other optimizers in the AntennaCAT chicken swarm family
        # use rng.choice([0,2]) here instead; swap the line below to match
        # the family behavior if preferred.

        mother_idx = int(self.chicken_info[particle][2]) # the idx of the mother chicken
        mother_loc = self.M[mother_idx]

        FL = self.rng.uniform(0, 2)
        self.M[particle] = np.round(self.M[particle] + FL*(mother_loc-self.M[particle]), self.number_decimals)


    def assign_subgroups(self):
        # CSO-PSO hybridization. 2022 ICSO, Sec. 3.2/3.3.
        # "The whole population is randomly divided into two parts,
        # namely, subgroup 1 and subgroup 2", with the same scales.
        # Subgroup 1 (pso_flags == False) moves by the (RE)CSO role rules.
        # Subgroup 2 (pso_flags == True) moves by the PSO velocity-position
        # update. This re-division happens at every iteration in the paper;
        # here, at the start of every full cycle through the population.
        # Because both subgroups share one population, one fitness record,
        # and one global best, the paper's 'merge subgroups to realize
        # information exchange' step is implicit in this framework.

        n_pso = self.number_of_particles // 2  # 'same scales'. if odd, the
                                               # extra particle stays in the
                                               # CSO subgroup
        shuffled_idx = self.rng.permutation(self.number_of_particles)
        self.pso_flags = np.zeros((self.number_of_particles), dtype=bool)
        self.pso_flags[shuffled_idx[0:n_pso]] = True


    def pso_update(self, particle):
        # CSO-PSO hybridization. 2022 ICSO, Sec. 3.2/3.3 (paper step 6).
        # This is the movement model for particles assigned to subgroup 2
        # for the current cycle. It REPLACES the CSO role move for those
        # particles (it is not a refinement applied after it).
        # Standard PSO velocity-position update using the particle's
        # personal best (Pb) and the swarm global best (Gb):
        #   V = W*V + C1*rand*(Pb - X) + C2*rand*(Gb - X)
        #   X = X + V
        # c1 = c2 = 2 per paper Sec. 4.1.2. NOTE: the paper does not
        # specify an inertia weight; the linearly decreasing weight
        # (W_max -> W_min over the run) is a framework choice kept for
        # consistency with the AntennaCAT optimizer family.

        self.W = self.W_max - (self.W_max - self.W_min)*(self.iter/np.maximum(self.maxit, 1))

        # personal bests are initialized to sys.maxsize. do not apply the
        # Pb/Gb learning terms until they hold real evaluated locations,
        # or the velocity will be driven by the placeholder values.
        if np.linalg.norm(self.F_Pb[particle]) >= sys.maxsize:
            return
        if np.linalg.norm(self.F_Gb) >= sys.maxsize:
            return

        r1 = self.rng.uniform(0, 1)
        r2 = self.rng.uniform(0, 1)

        term_pb = self.C1*r1*(self.Pb[particle] - self.M[particle])
        term_gb = self.C2*r2*(np.squeeze(self.Gb) - self.M[particle])

        # clip to match the overflow protections used in move_hen()
        self.V[particle] = np.clip(self.W*self.V[particle] + term_pb + term_gb, -10e30, 10e30)
        self.M[particle] = np.round(self.M[particle] + self.V[particle], self.number_decimals)


    def replication_elimination_dispersal(self):
        # RECSO operations. 2022 ICSO, Sec. 3.1. BFA-inspired.
        # Called at the START of reorganize_swarm(), BEFORE the swarm is
        # re-ranked (paper Sec. 3.3, step 5(b) precedes step 5(c)), so the
        # 'chicks' here are defined by the PREVIOUS role assignment. In
        # this framework the previous reorganization left the swarm sorted
        # best (idx 0) to worst (idx N-1), and indices do not move between
        # role updates, so the last CN entries are the chicks.
        #
        # 1) REPRODUCTION (paper Eq. 7):
        #       X_i(t+1) = X_(i-rNum-hNum)(t)
        #    the chick at index i inherits the CURRENT position of the
        #    individual at index i-rNum-hNum, i.e. the chicks collectively
        #    inherit the positions of the first CN (strongest) individuals.
        # 2) ELIMINATION-DISPERSAL (paper Eq. 8): each chick is then
        #    dispersed to a uniformly random in-bounds position with
        #    probability Ped (= 0.25 per paper Secs. 3.1.2 and 4.1.2).
        #
        # FRAMEWORK ADAPTATION NOTE: the paper's Eq. 7 copies position
        # only; this framework also copies the source's Pb/F_Pb (so the
        # fitness used by the re-ranking and movement formulas matches the
        # inherited position) and resets the Pb/F_Pb of dispersed chicks
        # (so they re-rank as unevaluated/worst and explore from the new
        # location instead of being pulled back by an inherited best).

        if self.CN < 1:
            return

        variation = self.ubound - self.lbound

        for i in range(0, self.CN):
            chick_idx = self.number_of_particles - self.CN + i
            source_idx = i  # i-th best individual at the last role update

            # REPRODUCTION: copy position, velocity, and personal best
            self.M[chick_idx] = 1*self.M[source_idx]
            self.V[chick_idx] = 1*self.V[source_idx]
            self.Pb[chick_idx] = 1*self.Pb[source_idx]
            self.F_Pb[chick_idx] = 1*self.F_Pb[source_idx]
            self.Active[chick_idx] = 1*self.Active[source_idx]

            # ELIMINATION-DISPERSAL with probability Ped
            # paper Eq. 8: X = lb + (ub - lb)*rand
            if self.rng.uniform(0, 1) < self.PED:
                self.M[chick_idx] = np.round(
                    np.squeeze(
                        np.multiply(self.rng.random((1, np.shape(self.M)[1])), variation)
                        + self.lbound
                    ), self.number_decimals)
                # reset velocity and personal best so the dispersed chick
                # explores from its new location
                self.V[chick_idx] = np.zeros(np.shape(self.M)[1])
                self.Pb[chick_idx] = sys.maxsize*np.ones(np.shape(self.M)[1])
                self.F_Pb[chick_idx] = sys.maxsize*np.ones(self.output_size)
                # a dispersed chick is placed in-bounds by Eq. 8, so it is
                # re-activated. this also gives INVISIBLE boundary (4) runs
                # a natural revival mechanism at each role update
                self.Active[chick_idx] = 1



    def reorganize_swarm(self):
        # 2022 ICSO: the reproduction & elimination-dispersal operations
        # (Sec. 3.1) are applied to the chicks BEFORE the swarm is
        # re-ranked, using the role labels from the PREVIOUS assignment
        # (paper Sec. 3.3: step 5(b) precedes step 5(c)). They are skipped
        # at the first role update, when fitness-based roles have not yet
        # been assigned (paper: 'we judge whether it is the first iteration
        # of the algorithm, if so, we go to step (c)').
        # NOTE: if CN > half the swarm, the source (best) and target
        # (chick) index ranges overlap and early replications can be
        # re-copied. Standard CSO population splits do not hit this case.
        if self.ran_reorganize:
            self.replication_elimination_dispersal()
        self.ran_reorganize = True

        # rank the chickens' fitness vals and establish hierarchial order
        # divide swarm into groups, determine relationship between mother hens and chicks

        #get the indexs that sort the personal best fitness from best (lowest) to worst
        l2_norm_vals = []
        for idx in range(0, self.number_of_particles):
            l2_norm_vals.append(np.linalg.norm(self.F_Pb[idx]))
      
        fitness_sort_idx = np.argsort(l2_norm_vals)# lowest are first
        
        #use the idx values to sort self.M and personal best fitness self.F_Pb
        temp_M = 1*self.M
        temp_F_Pb = 1*self.F_Pb
        temp_Active = 1*self.Active #tracking which are active
        ctr = 0

        for idx in fitness_sort_idx:
            temp_M[ctr] = self.M[idx]
            temp_F_Pb[ctr] = self.F_Pb[idx]
            temp_Active[ctr] = self.Active[idx]
            ctr = ctr + 1
        self.M = 1*temp_M
        self.F_Pb = 1*temp_F_Pb
        self.Active = 1*temp_Active

        # update the chicken_information (category, group, mother-child ID)
        # top RN are roosters. Middle HN are hens, with MN being mother hens.
        # last (and worst preforming) are chicks

        # make temp classification list
        classList = np.zeros(self.number_of_particles)
        # rooster
        classList[0:self.RN] = 0
        # hen
        start = self.RN
        end = self.RN + self.HN
        classList[start:end] = 1
        # mother hen
        start = self.RN + self.HN
        end = self.RN + self.HN + self.MN
        classList[start:end] = 2
        # chicks
        start = self.RN + self.HN + self.MN
        classList[start:] = 3
        # array of groups to use in random allocation
        group_nums = np.arange(self.RN)

        # first rooster, to reset the array
        self.chicken_info = np.array([0, 0, -1])

        for i in range(1,int(self.number_of_particles)): #start with the 2nd rooster/chicken
            if classList[i] == 0: #rooster
                # assign to the next group (i-1), and done.
                # CLASSIFICATION(0-4), GROUP(0-m), MOTHER-CHILD ID
                self.chicken_info = \
                    np.vstack([self.chicken_info, [classList[i], i, -1]])

            elif (classList[i] == 1) or (classList[i] == 2): #hen, mother hen
                # assign to a random group.
                # CLASSIFICATION(0-4), GROUP(0-m), MOTHER-CHILD ID
                hen_group = self.rng.choice(group_nums)
                self.chicken_info = \
                    np.vstack([self.chicken_info,[classList[i], hen_group, -1]])

            elif classList[i] == 3: #chick
                # select a random hen to be the 'mother' and assign to group
                # CLASSIFICATION(0-4), GROUP(0-m), MOTHER-CHILD ID
                groupAssigned = False
                while (groupAssigned == False):
                    chicken_idx = self.rng.integers(0, i-1, endpoint=False)# index after a chick will always be the chick
                    if self.chicken_info[chicken_idx][0] == 2: #is mother hen
                        # get the group the random mother hen is from
                        mother_group = self.chicken_info[chicken_idx][1]         
                        # assign chick to group, and to that mother hen                    
                        self.chicken_info = \
                            np.vstack([self.chicken_info,[classList[i], mother_group, chicken_idx]])

                        groupAssigned = True


  
    def check_bounds(self, particle):
        update = 0
        for i in range(0,(np.shape(self.M)[1])):
            if (self.lbound[i] > self.M[particle,i]) \
               or (self.ubound[i] < self.M[particle,i]):
                update = i+1        
        return update


    def random_bound(self, particle):
        # If particle is out of bounds, bring the particle back in bounds
        # The first condition checks if constraints are met, 
        # and the second determins if the values are to large (positive or negitive)
        # and may cause a buffer overflow with large exponents (a bug that was found experimentally)
        update = self.check_bounds(particle) or not self.constr_func(self.M[particle])
        if update > 0:
            while (self.check_bounds(particle) > 0) or (self.constr_func(self.M[particle]) == False):
                variation = self.ubound - self.lbound
                self.M[particle] = np.round(
                    np.squeeze(
                        self.rng.random() *
                        np.multiply(np.ones((1, np.shape(self.M)[1])), variation) +
                        self.lbound
                    ), self.number_decimals)
            
    def reflecting_bound(self, particle):
        # NOTE: chicken swarm has no velocity array (chickens move by position
        # rules), so unlike PSO there is no velocity to reflect. The position is
        # snapped back to the last in-bounds location.
        update = self.check_bounds(particle)
        constr = self.constr_func(self.M[particle])
        if (update > 0) and constr:
            self.M[particle] = 1*self.Mlast
        if not constr:
            self.random_bound(particle)

    def absorbing_bound(self, particle):
        # NOTE: chicken swarm has no velocity array, so there is no velocity to
        # absorb. The position is snapped back to the last in-bounds location.
        update = self.check_bounds(particle)
        constr = self.constr_func(self.M[particle])
        if (update > 0) and constr:
            self.M[particle] = 1*self.Mlast
        if not constr:
            self.random_bound(particle)

    def invisible_bound(self, particle):
        update = self.check_bounds(particle) or not self.constr_func(self.M[particle])
        if update > 0:
            self.Active[particle] = 0  
        else:
            pass          


    def handle_bounds(self, particle):
        if self.boundary == 1:
            self.random_bound(particle)
        elif self.boundary == 2:
            self.reflecting_bound(particle)
        elif self.boundary == 3:
            self.absorbing_bound(particle)
        elif self.boundary == 4:
            self.invisible_bound(particle)
        else:
            self.debug_message_printout("Error: No boundary is set!")

    def check_global_local(self, Flist, particle):

        if np.linalg.norm(Flist) < np.linalg.norm(self.F_Gb):
            self.F_Gb = np.array([Flist])
            self.Gb = np.array(self.M[particle])
        
        if np.linalg.norm(Flist) < np.linalg.norm(self.F_Pb[particle]):
            self.F_Pb[particle] = np.squeeze(Flist)
            self.Pb[particle] = self.M[particle]
    
    def converged(self):
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence
    
    def maxed(self):
        max_iter = self.iter >= self.maxit
        return max_iter
    
    def complete(self):
        done = self.converged() or self.maxed() or self.swarm_stalled
        return done
    
    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) +"\n" + \
                "-----------------------------\n" + \
                "Current Particle:\n" + \
                str(self.current_particle) +"\n" + \
                "Current Particle Active\n" + \
                str(self.Active[self.current_particle]) +"\n" + \
                "Current Particle Location\n" + \
                str(self.M[self.current_particle]) +"\n" + \
                "Absolute mean deviation\n" + \
                str(self.absolute_mean_deviation_of_particles()) +"\n" + \
                "-----------------------------"
            self.debug_message_printout(msg)
            
        if self.allow_update: # The first time step is called, this is false
            # safety stop: with INVISIBLE boundaries (4) it is possible for
            # every particle to leave the search space and go inactive. when
            # that happens no particle can move or be evaluated, self.iter
            # stops advancing, and a 'while not complete()' driver would
            # hang. flag the stall so complete() ends the run instead
            if not np.any(self.Active):
                if not self.swarm_stalled:
                    self.swarm_stalled = True
                    self.debug_message_printout("WARNING: all particles are inactive \
                                                (out of bounds with INVISIBLE boundary). Ending the optimization early.")
                return

            # 2022 ICSO: at the start of every full cycle through the
            # population, the swarm is randomly re-divided into two
            # equal-scale subgroups (Sec. 3.2/3.3): subgroup 1 moves by
            # the (RE)CSO role rules, subgroup 2 moves by the PSO
            # velocity-position update. this happens whether or not the
            # first particle is currently active
            if self.current_particle == 0:
                self.assign_subgroups()

            if self.Active[self.current_particle]:
                # save global best
                self.check_global_local(self.Flist,self.current_particle)

                # every self.G full cycle iterations:
                #           reorganize the swarm (includes the 2022 ICSO
                #           reproduction & elimination-dispersal operations
                #           applied BEFORE the re-ranking)
                if self.iter%self.G_steps == 0:
                    self.reorganize_swarm()
                    #start with the new best rooster
                    self.current_particle = 0
                    # roles changed and a new cycle starts: re-divide
                    self.assign_subgroups()
                
                if self.pso_flags[self.current_particle]: 
                    # PSO subgroup (paper step 6). the PSO update REPLACES
                    # the CSO role move for this particle this cycle
                    self.pso_update(self.current_particle)

                else:
                    # (RE)CSO subgroup (paper step 5)
                    # move chickens
                    # roosters are always at the top of the list so that they're moved first.
                    # Then the hens are moved. It doesn't matter which type of hen is moved first.
                    # Chicks are moved last so that they can follow the mother hens
                    chicken_type = self.chicken_info[self.current_particle][0]
                    if chicken_type == 0: #update rooster location
                        self.move_rooster(self.current_particle)

                    elif (chicken_type == 1): #update hen location
                        self.move_hen(self.current_particle)

                    elif (chicken_type == 2): #update mother hen location
                        self.move_hen(self.current_particle)

                    elif chicken_type == 3: #update chick location
                        self.move_chick(self.current_particle)

                # handle any out-of-bounds situation
                self.handle_bounds(self.current_particle)

            self.current_particle = self.current_particle + 1
            if self.current_particle == self.number_of_particles:
                self.current_particle = 0

            if self.complete() and not suppress_output:
                msg =  "\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.debug_message_printout(msg)

    def export_swarm(self):
        #These do NOT export.
        # # These are passed objects created at runtim
        # self.parent # this is an object in memory at runtime
        # self.surrogateOptimizer =  # this is an object in memory at runtime  
        # self.obj_func =  # this is an object in memory at runtime                                             
        # self.constr_func =  # this is an object in memory at runtime    
        # self.useSurrogateModel = # this NEEDS to match every time. Should be part of the init() 
        # self.number_decimals = # this can be changed. IT might be interesting to change between runs
        # self.boundary = boundary     # int. can be chaged, but needs a default
        # These export:


        swarm_export = {            
            # These are values that define the swarm and current solution space
            # These are retained because the dimensionality of M, F_pb, etc. are strict
            'evaluate_threshold': [self.evaluate_threshold],
            'obj_threshold': [self.obj_threshold],
            'targets': [self.targets],
            'lbound': [self.lbound],
            'ubound': [self.ubound],
            'output_size': [self.output_size], # this can be calculated if needed
            # convergence and step criteria
            'maxit': [self.maxit],                                       
            'E_TOL': [self.E_TOL],                                            
            'iter': [self.iter],
            'current_particle': [self.current_particle],    
            'allow_update': [self.allow_update],
            # optimizer specfic
            'RN': [self.RN],
            'HN': [self.HN],
            'MN': [self.MN],
            'CN': [self.CN],
            'G': [self.G],
            'W_min': [self.W_min],
            'W_max': [self.W_max],
            'W': [self.W],
            'C1': [self.C1],
            'C2': [self.C2],
            'PED': [self.PED],
            'V': [self.V],
            'pso_flags': [self.pso_flags],
            'ran_reorganize': [self.ran_reorganize],
            'swarm_stalled': [self.swarm_stalled],
            'number_of_particles': [self.number_of_particles], 
            # shared format vars for AntennaCAT set
            'M': [self.M], 
            'Active': [self.Active],                    
            'Gb': [self.Gb],
            'F_Gb': [self.F_Gb],             
            'Pb': [self.Pb],           
            'F_Pb': [self.F_Pb],
            'Flist': [self.Flist],                                                
            'Fvals': [self.Fvals],                                               
            'Mlast': [self.Mlast]
            } 
        
        return swarm_export # this is turned into a dataframe in the driver class

    def import_swarm(self, swarm_export):
        
        # swarm export is a dataframe. this is unpacked and converted just like
        # with the initialized opt_df params

        # These are values that define the swarm and current solution space
        # These are retained because the dimensionality of M, F_pb, etc. are strict
        self.evaluate_threshold = bool(swarm_export['evaluate_threshold'][0]) 
        self.obj_threshold = np.array(swarm_export['obj_threshold'][0]) 
        self.targets = np.array(swarm_export['targets'][0]).reshape(-1, 1)   

        self.lbound = np.array(swarm_export['lbound'][0]) 
        self.ubound = np.array(swarm_export['ubound'][0]) 
        self.output_size = int(swarm_export['output_size'][0])  # this can be calculated if needed
        # convergence and step criteria
        self.maxit = int(swarm_export['maxit'][0])                                              
        self.E_TOL = float(swarm_export['E_TOL'][0])                                               
        self.iter = int(swarm_export['iter'][0])     # NEED 'RESUME' and 'START OVER' options
        self.current_particle = int(swarm_export['current_particle'][0])         
        self.allow_update = int(swarm_export['allow_update'][0])    # BOOL as INT

        # optimizer specfic
        self.RN = int(swarm_export['RN'][0]) 
        self.HN = int(swarm_export['HN'][0]) 
        self.MN = int(swarm_export['MN'][0]) 
        self.CN = int(swarm_export['CN'][0]) 
        self.G = int(swarm_export['G'][0]) 
        self.W_min = float(swarm_export['W_min'][0]) 
        self.W_max = float(swarm_export['W_max'][0]) 
        self.W = float(swarm_export['W'][0]) 
        self.C1 = float(swarm_export['C1'][0]) 
        self.C2 = float(swarm_export['C2'][0]) 
        self.PED = float(swarm_export['PED'][0]) 
        self.V = np.array(swarm_export['V'][0]) 
        self.pso_flags = np.array(swarm_export['pso_flags'][0], dtype=bool) 
        self.ran_reorganize = bool(swarm_export['ran_reorganize'][0]) 
        self.swarm_stalled = bool(swarm_export['swarm_stalled'][0]) 
        self.number_of_particles = int(swarm_export['number_of_particles'][0]) 

        # shared format vars for AntennaCAT set

        self.M = np.array(swarm_export['M'][0]) 
        self.Active = np.array(swarm_export['Active'][0])                    
        self.Gb = np.array(swarm_export['Gb'][0]) 
        self.F_Gb = np.array(swarm_export['F_Gb'][0])
        self.Pb = np.array(swarm_export['Pb'][0])              
        self.F_Pb = np.array(swarm_export['F_Pb'][0])  
        self.Flist = np.array(swarm_export['Flist'][0])                                                 
        self.Fvals= np.array(swarm_export['Fvals'][0])                                               
        self.Mlast= np.array(swarm_export['Mlast'][0])    

    def get_obj_inputs(self):
        return np.vstack(self.M[self.current_particle])
    
    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        iteration = 1*self.iter
        return iteration, best_eval
        
    def get_optimized_soln(self):
        return self.Gb.reshape(-1, 1) #standardization  
    
    def get_optimized_outs(self):
        return self.F_Gb[0] #correction for extra brackets that happen with the math/passing
    
    def absolute_mean_deviation_of_particles(self):
        mean_data = np.array(np.mean(self.M, axis=0)).reshape(1, -1)
        abs_data = np.zeros(np.shape(self.M))
        for i in range(0,self.number_of_particles):
            abs_data[i] = np.squeeze(np.abs(self.M[i]-mean_data))

        abs_mean_dev = np.linalg.norm(np.mean(abs_data,axis=0))
        return abs_mean_dev

    def debug_message_printout(self, msg):
        if self.parent == None:
            print(msg)
        else:
            self.parent.debug_message_printout(msg)