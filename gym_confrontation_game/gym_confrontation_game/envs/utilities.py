import numpy as np
import random
import torch
import dgl

def spa_deriv(index, V, g, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension
    From Michael
    Args:
        index: (a1x, a1y)
        V (ndarray): [..., neg2pos] where neg2pos is a list [scalar] or []
        g (class): the instance of the corresponding Grid
        periodic_dims (list): the corrsponding periodical dimensions []

    Returns:
        List of left and right spatial derivatives for each dimension
    """
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1:])

        next_index = tuple(
            left_index + [index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [index[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [V.shape[dim] - 1] + right_index
                )
                left_boundary = V[left_periodic_boundary_index]
            else:
                left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = V[right_periodic_boundary_index]
            else:
                right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2)[0])
    return spa_derivatives

def find_sign_change1v0(grid1v0, value1v0, current_state):
    """Return two positions (neg2pos, pos2neg) of the value function

    Args:
    grid1v0 (class): the instance of grid
    value1v0 (ndarray): including all the time slices, shape = [100, 100, len(tau)]
    current_state (tuple): the current state of the attacker
    """
    current_slices = grid1v0.get_index(current_state)
    current_value = value1v0[current_slices[0], current_slices[1], :]  # current value in all time slices
    neg_values = (current_value<=0).astype(int)  # turn all negative values to 1, and all positive values to 0
    checklist = neg_values - np.append(neg_values[1:], neg_values[-1])
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checklist==1)[0], np.where(checklist==-1)[0]

def compute_control1v0(agents_1v0, grid1v0, value1v0, tau1v0, position, neg2pos):
    """Return the optimal controls (tuple) of the attacker
    Notice: calculate the spatial derivative vector in the game
    Args:
    agents_1v0 (class): the instance of 1v0 attacker defender
    grid1v0 (class): the instance of grid
    value1v0 (ndarray): 1v0 HJ reachability value function with all time slices
    tau1v0 (ndarray): all time indices
    current_state (tuple): the current state of the attacker
    x1_1v0 (ndarray): spatial derivative array of the first dimension
    x2_1v0 (ndarray): spatial derivative array of the second dimension
    """
    assert value1v0.shape[-1] == len(tau1v0)  # check the shape of value function

    # check the current state is in the reach-avoid set
    current_value = grid1v0.get_value(value1v0[..., 0], list(position))
    if current_value > 0:
        value1v0 = value1v0 - current_value
    
    # calculate the derivatives
    v = value1v0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
    # print(f"The shape of the input value function v of attacker is {v.shape}. \n")
    spat_deriv_vector = spa_deriv(grid1v0.get_index(position), v, grid1v0)
    # print(f"The calculation of 2D spatial derivative vector is {end_time-start_time}. \n")
    return agents_1v0.optCtrl_inPython(spat_deriv_vector)


def attackers_control_1v0(agents_1v0, grid1v0, value1v0, tau1v0, current_attackers, stops_index):
    """Return a list of 2-dimensional control inputs of all attackers based on the value function
    Notice: calculate the spatial derivative vector in the game
    Args:
    agents_1v0 (class): the instance of 1v0 attacker defender
    grid1v0 (class): the corresponding Grid instance
    value1v0 (ndarray): 1v0 HJ reachability value function with all time slices
    tau1v0 (ndarray): all time indices
    agents_1v0 (class): the corresponding AttackerDefender instance
    current_positions (list): the attacker(s), [(), (),...]
    x1_1v0 (ndarray): spatial derivative array of the first dimension
    x2_1v0 (ndarray): spatial derivative array of the second dimension
    """
    control_attackers = []
    for i in range(len(current_attackers)):
        position = current_attackers[i]
        neg2pos, pos2neg = find_sign_change1v0(grid1v0, value1v0, position)
        # print(f"The neg2pos is {neg2pos}.\n")
        if len(neg2pos):
            if i in stops_index:
                control_attackers.append((0.0, 0.0))
            else:
                control_attackers.append(compute_control1v0(agents_1v0, grid1v0, value1v0, tau1v0, position, neg2pos))
        else:
            control_attackers.append(((0.5, 0.5)-np.array(position))/np.linalg.norm((0.5, 0.5)-np.array(position)))
    return control_attackers

def next_positions(current_positions, controls, tstep):
    """Return the next positions (list) of attackers or defenders

    Arg:
    current_positions (list): [(), (),...]
    controls (list): [(), (),...]
    """
    temp = []
    num = len(controls)
    for i in range(num):
        temp.append((current_positions[i][0]+controls[i][0]*tstep, current_positions[i][1]+controls[i][1]*tstep))
    return temp

def distance(attacker, defender):
    """Return the 2-norm distance between the attacker and the defender

    Args:
    attacker (tuple): the position of the attacker
    defender (tuple): the position of the defender
    """
    d = np.sqrt((attacker[0]-defender[0])**2 + (attacker[1]-defender[1])**2)
    return d

def get_connectivity(x, comm_rad2):
    n_defender = x.shape[0]
    x_loc = np.reshape(x, (n_defender,2,1))
    dist_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
    np.fill_diagonal(dist_net, np.Inf)
    a_net = (dist_net < comm_rad2).astype(float)

    return a_net

def build_graph(pos_list, node_feature, stopped_agents, comm_rad):
    if len(pos_list) != len(node_feature):
        print("Error: Length of pos_list and node_feature are not the same.")
    if len(pos_list) == 0:
        print("Error: Length of pos_list is 0.")
    pos_array = np.vstack(pos_list)
    edge_feat_list = []
    g=dgl.graph([])
    num_ag = len(pos_list)

    g.add_nodes(num_ag)
    adj_matrix = get_connectivity(pos_array, comm_rad*comm_rad)
    edge_list = []
    for i in range(0,num_ag):
        # add self loop for all valid agents
        edge_list.append((i,i))
        edge_feat_list.append(0)
        if i in stopped_agents:
            continue
        for j in range(0,num_ag):
            if j in stopped_agents:
                continue
            if adj_matrix[i][j] > 0:
                edge_list.append((i,j))
                edge_feat_list.append(np.sqrt(np.square(pos_array[j][0]-pos_array[i][0])+np.square(pos_array[j][1]-pos_array[i][1])))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    g.set_e_initializer(dgl.init.zero_initializer)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata['feat'] = torch.stack(node_feature)
    # edge_weight = 1/torch.tensor(edge_feat_list+edge_feat_list)
    # g.edata['weight'] = (edge_weight - torch.min(edge_weight)) / (torch.max(edge_weight) - torch.min(edge_weight))

    return g