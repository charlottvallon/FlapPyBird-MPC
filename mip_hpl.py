import cvxpy as cvx
import numpy as np
import joblib
import matplotlib.pyplot as plt



N = 24 # time steps to look aheadN
path = cvx.Variable((N, 2)) # initialize the y pos and y velocity
flap = cvx.Variable(N-1, boolean=True) # initialize the inputs, whether or not the bird should flap in each step

PIPEGAPSIZE  = 100 # gap between upper and lower pipe
PIPEWIDTH = 52
BIRDWIDTH = 34
BIRDHEIGHT = 24
BIRDDIAMETER = np.sqrt(BIRDHEIGHT**2 + BIRDWIDTH**2) # the bird rotates in the game, so we use it's maximum extent
SKY = 0 # location of sky
GROUND = (512*0.79)-1 # location of ground
PLAYERX = 57 # location of bird


def getPipeConstraintsDistance(x, y, lowerPipes):
    constraints = [] # init pipe constraint list
    pipe_dist = 0 # init dist from pipe center
    for pipe in lowerPipes:
        dist_from_front = pipe['x'] - x - BIRDDIAMETER
        dist_from_back = pipe['x'] - x + PIPEWIDTH
        if (dist_from_front < 0) and (dist_from_back > 0):
            constraints += [y <= (pipe['y'] - BIRDDIAMETER)] # y above lower pipe
            constraints += [y >= (pipe['y'] - PIPEGAPSIZE + 10)] # y below upper pipe (added birddiameter)
            pipe_dist += cvx.abs(pipe['y'] - (PIPEGAPSIZE//2) - (BIRDDIAMETER//2) - y) # add distance from center
    return constraints, pipe_dist


def solve(playery, playerVelY, lowerPipes, prev_flaps, prev_path, est_y = 0, std_y = 100):
    pipeVelX = -4 # speed in x
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -14   # players speed on flapping

    # unpack path variables
    y = path[:,0]
    vy = path[:,1]
    c = [] # init constraint list
    c += [y <= GROUND, y >= SKY] # constraints for sky and ground
    c += [y[0] == playery, vy[0] == playerVelY] # initial conditions

    obj = 0

    x = PLAYERX
    xs = [x] # init x list
    for t in range(N-1): # look ahead
        dt = t//15 + 1 # let time get coarser further in the look ahead
        x -= dt * pipeVelX # update x
        xs += [x] # add to list
        c += [vy[t + 1] ==  vy[t] + playerAccY * dt + playerFlapAcc * flap[t] ] # add y velocity constraint, f=ma
        c += [y[t + 1] ==  y[t] + vy[t + 1]*dt ] # add y constraint, dy/dt = a
        pipe_c, dist = getPipeConstraintsDistance(x, y[t+1], lowerPipes) # add pipe constraints
        c += pipe_c
        obj += dist
        
    # New code for terminal constraint
    # Pipe Region:
    if lowerPipes[-1]['x'] < 181 and 181 < lowerPipes[-1]['x'] + 52:
        c += [y[t+1] >= (lowerPipes[-1]['y']-100) + cvx.square(vy[t + 1])/2]
    else:
        pass    
    # Btwn-Pipe Region
    if lowerPipes[-1]['x'] > 181:
        n = abs((lowerPipes[-1]['x'] - 181)/4)
        c += [y[t+1] >= (lowerPipes[-1]['y']-100) - n*(n-1)/2 - vy[t+1]*n] 
        pass

    # Add c1 and check if terminal set within gpr stddev is feasible
    c1 = c + [y[t+1] >= est_y-std_y] + [y[t+1] <= est_y+std_y] 

    objective = cvx.Minimize(cvx.sum(cvx.abs(vy)) + 100* obj)

    if std_y < 50:
        prob = cvx.Problem(objective, c1) # if low std, use the strategy
    else:
        prob = cvx.Problem(objective, c) # if high standard deviation, do not solve with the strategy    
    
    try: # try solving problem with c1, otherwise try using c
        prob.solve(verbose = False, solver="GUROBI") 
        new_path = list(zip(xs, y.value)) # store the path
        new_flaps = np.round(flap.value).astype(bool) # store the solution
        
        return new_flaps, new_path # return the one-step flap, and the entire path    
    except:      
        try:
            prob = cvx.Problem(objective, c)
            prob.solve(verbose = False, solver="GUROBI")
            new_path = list(zip(xs, y.value)) # store the path
            new_flaps = np.round(flap.value).astype(bool) # store the solution
            
            return new_flaps, new_path # return the one-step flap, and the entire path            
        
        except:
            new_flaps = prev_flaps[1:] # if we didn't get a solution this round, use the inputs and flaps from the last solve iter
            new_path = [((x-4), y) for (x,y) in prev_path[1:]]
            
            if len(prev_flaps)<12:
                #print('returning false')
                return [False], [(0,0), (0,0)]
            else: 
                #print('is this what is causing the error')
                return new_flaps, new_path


