import pandas as pd
import numpy as np
import random
import time
from plots import *
from copy import deepcopy
import time
from plots import *
import sys
from queue import PriorityQueue

plot_flag = sys.argv[1]
map = sys.argv[2]
part = sys.argv[3]


def value_update(m,n,map, x , y ,p, gamma,living_reward, V, policy):

    actions = [ 'r', 'l', 't', 'd']

    dx = x
    dy = y
    reward_total = {}
    s = 0
    for  a in actions:
        if a == 't':
            dx = max(x-1, 0)
            dy = y
        if a == 'd':
            dx = min(x+1, n-1)
            dy = y
        if a == 'l':
            dx = x
            dy = max(y-1, 0)
        
        if a == 'r':
            dx = x
            dy = min(y+1, m-1)
        reward = 0
        if map[dx][dy]=='F' or map[dx][dy]=='S':
            reward = living_reward
        reward_total[a] = reward +gamma*V[dx][dy]
        s+=reward_total[a]

    t = (1-p)/3
    max_value = -1*np.inf
    for a in actions:
        value = p*reward_total[a]+t*(s-reward_total[a])
        if value > max_value:
            V[x][y] = value
            max_value = value
            policy[x][y] = a

    return V[x][y], policy[x][y]

def value_update_transition(m,n,map, x , y, gamma,living_reward, V, policy):

    actions = [ 'r', 'l', 't', 'd']

    dx = x
    dy = y
    reward_total = {}
    s = 0
    for  a in actions:
        if a == 't':
            dx = max(x-1, 0)
            dy = y
        if a == 'd':
            dx = min(x+1, n-1)
            dy = y
        if a == 'l':
            dx = x
            dy = max(y-1, 0)
        
        if a == 'r':
            dx = x
            dy = min(y+1, m-1)
        reward = 0
        if map[dx][dy]=='F' or map[dx][dy]=='S':
            reward = living_reward
        reward_total[a] = reward +gamma*V[dx][dy]
        s+=reward_total[a]

    max_value = -1*np.inf
    for a in actions:
        if a =='r':
            value = 1/3*reward_total[a]+1/3*reward_total['t']+1/3*reward_total['d']
        if a =='l':
            value = 1/3*reward_total[a]+1/3*reward_total['t']+1/3*reward_total['d']
        if a =='t':
            value = 1/3*reward_total[a]+1/3*reward_total['r']+1/3*reward_total['l']
        if a =='d':
            value = 1/3*reward_total[a]+1/3*reward_total['r']+1/3*reward_total['l']
        if value > max_value:
            V[x][y] = value
            max_value = value
            policy[x][y] = a

    return V[x][y], policy[x][y]

def value_update_policy(m,n, x , y ,p, gamma, V, action):

    actions = [ 'r', 'l', 't', 'd']

    dx = x
    dy = y
    reward = {}
    s = 0
    for  a in actions:
        if a == 't':
            dx = max(x-1, 0)
            dy = y
        if a == 'd':
            dx = min(x+1, n-1)
            dy = y
        if a == 'l':
            dx = x
            dy = max(y-1, 0)
        
        if a == 'r':
            dx = x
            dy = min(y+1, m-1)
        reward[a] = gamma*V[dx][dy]
        s+=reward[a]

    t = (1-p)/3

    value = p*reward[action]+t*(s-reward[action])


    return value


    
    
def policy_improvement(m, n, x , y ,p, gamma, V, action, value):
    actions = [ 'r', 'l', 't', 'd']
    actions.remove(action)

    dx = x
    dy = y
    reward = {}
    s = 0
    for  a in actions:
        if a == 't':
            dx = max(x-1, 0)
            dy = y
        if a == 'd':
            dx = min(x+1, n-1)
            dy = y
        if a == 'l':
            dx = x
            dy = max(y-1, 0)
        
        if a == 'r':
            dx = x
            dy = min(y+1, m-1)
        reward[a] = gamma*V[dx][dy]
        s+=reward[a]

    t = (1-p)/3
    value_dict = {}
    for a in actions:
        value_new = p*reward[a]+t*(s-reward[a])
        value_dict[a]=value_new

    max_act = max(value_dict, key=value_dict.get)
    max_value = value_dict[max_act]
    if max_value > value:
        action = max_act

    return action

def get_best_policy(map, policy, p, gamma, V):
    m = len(map)
    n = len(map[0])
    for i in range(len(policy)):
        for j in range(len(V)):
            if map[i][j]=='G' or map[i][j]=='H':
                continue
            else:
                action = policy_improvement(m,n,i,j,p,gamma,V,policy[i][j], V[i][j])
                policy[i][j] = action

    return policy

def policy_evaluate(map, p, gamma, policy, epsilon):
    m = len(map[0])
    n = len(map)
    V = [[0 for j in range(m)] for i in range(n)]
    x_goal, y_goal = np.where(map=='G')
    V[x_goal[0]][y_goal[0]] = 1
    flag = True
    while flag:
        V_old = deepcopy(V)
        for i in range(len(V)):
            for j in range(len(V[0])):
                if map[i][j] == 'G' or map[i][j]=='H':
                    continue
                else:
                    value = value_update_policy(m,n,i,j,p,gamma,V,policy[i][j])
                    V[i][j]= value
        diff = get_diff(V_old, V)
        if diff < epsilon:
            break
    
    return V

def get_diff(V_old, V):
    diff = 0 
    for i in range(len(V)):
        for j in range(len(V[0])):
            diff+=abs(V[i][j]-V_old[i][j])
    
    return diff

if __name__=='__main__':
    map_data = pd.read_csv(map, header=None)
    map = np.array(map_data)

    if part == 'value_iteration':
        V = np.array([[0.0 for j in range(len(map[0]))] for i in  range(len(map))])

        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 'G':
                    V[i][j]=1.0
                if map[i][j] == 'F':
                    # print(map[i][j])
                    V[i][j] = 0
                if map[i][j]=='S':
                    V[i][j] = 0
        m = len(map)
        n = len(map[0])
        num_iteration = 1
        print(map)

        num_valid = 0
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j]=='G' or map[i][j]=='H':
                    continue
                else:
                    num_valid+=1


        policy = [['' for j in range(len(V[0]))] for i in range(len(V))]


        epsilon = 0.001
        count = 0
        t1 = time.time()
        flag = True
        count = 0
        start_state = []
        while flag:
            V_old = deepcopy(V)
            for i in range(len(V)):
                for j in range(len(V[0])):
                    if map[i,j]=='H':
                        policy[i][j] = '_'
                        continue
                    if map[i,j]=='G':
                        policy[i][j] = '_'
                        continue
                    else:
                        value, action = value_update(m, n, map, i, j, 0.8, 0.9,0, V, policy)
                        
                        V[i][j]= value
                        policy[i][j]= action
                        if map[i][j]=='S':
                            start_state.append(value)

            diff = 0 
            for i in range(len(V)):
                for j in range(len(V[0])):
                    diff+=abs(V[i][j]-V_old[i][j])

            count+=1
            print(f'running iteration',{count})

            if diff<epsilon:
                print(diff)
                flag = False


        t2 = time.time()
        print(V)
        print(policy)
        print('total number of iterations are', count)
        print('total number of updates is ', count*num_valid)
        print('time taken is ', t2-t1)
        if plot_flag ==  'small':
            Plotting_small(V, policy)
        else:
            Plotting(V,policy)
    if part == 'Priority':
        priority_update = PriorityQueue()
        dx = [1,0,-1,0]
        dy = [-1,0,1,0]
        goal_index =  [(i, j) for i, row in enumerate(map) for j, element in enumerate(row) if element == 'G']
        x_goal, y_goal = goal_index[0]
        flag = True 
        priority_update.put((-1,x_goal,y_goal))
        V = np.array([[0.0 for j in range(len(map[0]))] for i in  range(len(map))])

        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 'G':
                    V[i][j]=1.0
                if map[i][j] == 'F':
                    # print(map[i][j])
                    V[i][j] = 0
                if map[i][j]=='S':
                    V[i][j] = 0
        num_valid = 0
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j]=='G' or map[i][j]=='H':
                    continue
                else:
                    num_valid+=1

        print(num_valid)
        m = len(map)
        print(m)
        n = len(map[0])
        print(n)
        num_iteration = 1
        policy = [['' for j in range(len(V[0]))] for i in range(len(V))]
        count = 0
        epsilon = 1e-9
        flag = True
        updates = 0
        iterations = 0
        V_old = deepcopy(V)
        start_state = []
        t1 = time.time()
        while flag: 
            error, x, y = priority_update.get()
            total = iterations*num_valid

            for x_ind in dx:
                for y_ind in dy:
                    x_new = x+x_ind
                    y_new = y+y_ind
                    
                    if (x_new < 0 or x_new>n-1 or y_new<0 or y_new>n-1):
                        continue
                    elif map[x_new][y_new]=='G' or map[x_new][y_new] == 'H':
                        policy[x_new][y_new]='_'
                        continue
                    else:

                        e1 = V[x_new][y_new]

                        value, action = value_update(m, n, map, x_new, y_new, 0.8, 0.9,0,0, V, policy)
                        V[x_new][y_new]= value
                        policy[x_new][y_new]= action
                        error = abs(e1-V[x_new][y_new])
                        priority_update.put((-1*error, x_new, y_new))
                        updates+=1    
                
            if updates>total:
                iterations+=1
                x_start,y_start = np.where(map=='S')
                start_state.append(V[x_start[0],y_start[0]])
                start_state.append
                diff = get_diff(V_old,V) 
                if diff<epsilon:
                        print(diff)
                        flag = False
                V_old = deepcopy(V)
                
        t2 = time.time()
        print(V)
        print(policy)
        print('number of iterations is ', iterations)
        print('total number of updates is', updates)
        print('total time taken is', t2-t1)
        print(start_state)
        if plot_flag == 'small':
            Plotting_small(V, policy)
        else:
            Plotting(V,policy)

    if part == 'policy':
        m = len(map)
        n = len(map[0])
        actions =  ['r','l','t','d']

        policy = [['_' for j in range(len(map[0]))] for i in range(len(map))]
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 'H' or map[i][j] == 'G':
                    policy[i][j] = '_'
                else:
                    policy[i][j] = random.choice(actions)

        V = [[0 for j in range(len(map[0]))] for i in range(len(map))]

        for i in range(len(V)):
            for j in range(len(V[0])):
                if map[i][j]=='G':
                    V[i][j] = 1
        t1 = time.time()
        epsilon = 1e-6
        iterations = 0    
        V =  policy_evaluate(map,0.8,0.9,policy, epsilon)
        policy = get_best_policy(map,policy,0.8,0.9,V)
        iterations = 1
        flag = True
        x_start, y_start = np.where(map=='S')
        start_state = []
        start_state.append(V[[x_start[0]][y_start[0]]][0])
        while flag:
            V_old = deepcopy(V)
            V = policy_evaluate(map,0.8,0.9,policy,epsilon)
            start_state.append(V[[x_start[0]][y_start[0]]][0])
            policy = get_best_policy(map,policy,0.8,0.9,V)
            diff = get_diff(V_old, V)
            print(diff)
            iterations+=1
            print('running iteration',iterations)
            if diff<epsilon:
                break
        t2 = time.time()
        print(t2-t1)
        print(V)
        print(iterations)
        print(policy)
        print(start_state)
        print(len(start_state))
        if plot_flag == 'small':
            Plotting_small(V, policy)
        else:
            Plotting(V, policy)

    if part == 'transition_analysis':
        
        V = np.array([[0.0 for j in range(len(map[0]))] for i in  range(len(map))])

        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 'G':
                    V[i][j]=1.0
                if map[i][j] == 'F':
                    # print(map[i][j])
                    V[i][j] = 0
                if map[i][j]=='S':
                    V[i][j] = 0
        m = len(map)
        n = len(map[0])
        num_iteration = 1
        print(map)

        num_valid = 0
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j]=='G' or map[i][j]=='H':
                    continue
                else:
                    num_valid+=1


        policy = [['' for j in range(len(V[0]))] for i in range(len(V))]


        epsilon = 0.001
        count = 0
        t1 = time.time()
        flag = True
        count = 0
        start_state = []
        while flag:
            V_old = deepcopy(V)
            for i in range(len(V)):
                for j in range(len(V[0])):
                    if map[i,j]=='H':
                        policy[i][j] = '_'
                        continue
                    if map[i,j]=='G':
                        policy[i][j] = '_'
                        continue
                    else:
                        value, action = value_update_transition(m, n, map, i, j, 0.9,0, V, policy)
                        V[i][j]= value
                        policy[i][j]= action
                        if map[i][j]=='S':
                            start_state.append(value)

            diff = 0 
            for i in range(len(V)):
                for j in range(len(V[0])):
                    diff+=abs(V[i][j]-V_old[i][j])

            count+=1
            print(f'running iteration',{count})

            if diff<epsilon:
                print(diff)
                flag = False


        t2 = time.time()
        print(V)
        print(policy)
        print('total number of iterations are', count)
        print('total number of updates is ', count*num_valid)
        print('time taken is ', t2-t1)
        if plot_flag == 'small':
            Plotting_small(V, policy)
        else:
            Plotting(V, policy)


    









