#  coding: UTF-8  #
"""
@filename: ACA_final.py
@author: Yingkai
@time: 2024-07-14
"""

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ******************************  Generate granular balls  *******************************
class GenerateGBs:
    def __init__(self, data, k):
        self.data = data[['x', 'y']]
        self.gbs_list = [self.data.index.tolist()]
        self.gbs_center = []
        self.gbs_radius = []
        # threshold for controlling the size of the granular ball.
        # k ∈ [0, 1]
        self.threshold = k * np.sqrt(len(self.data))

    def _gb_center_radius(self, gb):
        '''
        Calculate the center and radius of the gbs
        '''
        x_center = self.data.loc[gb, 'x'].mean()
        y_center = self.data.loc[gb, 'y'].mean()
        gb_center = (x_center, y_center)
        gb_radius = max(np.sqrt((self.data.loc[gb, 'x'] - x_center) ** 2 + (self.data.loc[gb, 'y'] - y_center) ** 2))
        return gb_center, gb_radius

    def _split_gb(self, gb_idx):
        '''
        Use K-Means to split a ball into two sub-balls.
        '''
        split_k = min(2, len(gb_idx))
        gb = self.data.loc[gb_idx]
        kmeans = KMeans(n_clusters=split_k, random_state=0).fit(gb)
        labels = kmeans.labels_

        sub_balls = []
        for single_label in range(split_k):
            sub_ball = [idx for idx, label in zip(gb_idx, labels) if label == single_label]
            sub_balls.append(sub_ball)
        return sub_balls

    def split(self):
        '''
        Split all balls if they have more points than a threshold.
        '''
        gb_list_new = []
        for gb_idx in self.gbs_list:
            if isinstance(gb_idx, int):
                gb_idx = [gb_idx]
            if len(gb_idx) < self.threshold:
                gb_list_new.append(gb_idx)
            else:
                gb_list_new.extend(self._split_gb(gb_idx))
        return gb_list_new

    def generate_gbs(self):
        '''
        Split the ball iteratively until no new splits are generated
        '''
        while True:
            gbs_num = len(self.gbs_list)
            self.gbs_list = self.split()
            gbs_num_new = len(self.gbs_list)
            if gbs_num == gbs_num_new:
                print('break, all GBs have been generated')
                break
        return self.gbs_list

    def assign_single_point_to_gbs(self):
        '''
        Assigning a ball containing a single point to the nearest multi-point ball
        :return: self.gbs_list
        '''
        gbs_center, gbs_radius = [], []
        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            gbs_center.append(center)
            gbs_radius.append(radius)
        gbs_center = np.array(gbs_center)

        new_merged_gbs = {}
        single_point_gbs = []

        # Separate single point GBs and multi-point GBs
        for gb in self.gbs_list:
            if len(gb) == 1:
                single_point_gbs.append(gb[0])
            else:
                label = len(new_merged_gbs)
                new_merged_gbs[label] = gb

        # Merge single point GBs into nearest multi-point GBs
        for sp in single_point_gbs:
            min_distance = float('inf')
            closest_gb_label = None
            sp_center = self.data.loc[sp, ['x', 'y']].values

            for label, gbs in new_merged_gbs.items():
                gb_center = self.data.loc[gbs, ['x', 'y']].mean().values
                distance = np.linalg.norm(sp_center - gb_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_gb_label = label

            if closest_gb_label is not None:
                new_merged_gbs[closest_gb_label].append(sp)

        self.gbs_list = list(new_merged_gbs.values())
        return self.gbs_list

    def plot_merge_gbs(self, merged_gbs, centers_radii, plt_type=0):
        plt.figure()
        plt.axis()
        for i in range(len(merged_gbs)):
            # Plot points in each GB as black dots
            if plt_type == 0:
                plt.plot(self.data.loc[merged_gbs[i], 'x'], self.data.loc[merged_gbs[i], 'y'], '.', c='k')

            # Plot circles and their centers
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = centers_radii[i][0][0] + centers_radii[i][1] * np.cos(theta)
                y = centers_radii[i][0][1] + centers_radii[i][1] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)
                center_marker = 'x' if plt_type == 0 else '.'
                plt.plot(centers_radii[i][0][0], centers_radii[i][0][1], center_marker, color='r')
                # Add index label near the circle center
                plt.text(centers_radii[i][0][0], centers_radii[i][0][1], str(i), color='blue', fontsize=12, ha='right')

        plt.show()

    def merge_contained_gbs(self, merged_gbs):
        '''
        Merge contained balls, i.e., when one ball is completely contained within another ball, combine them
        '''
        # Compute centers and radii for all GBs
        centers_radii = {label: self._gb_center_radius(gbs) for label, gbs in merged_gbs.items()}

        # Identify containment relationships
        containment_relations = {}
        for label, (center, radius) in centers_radii.items():
            for other_label, (other_center, other_radius) in centers_radii.items():
                if label != other_label:
                    distance = np.linalg.norm(np.array(center) - np.array(other_center))
                    if distance + radius <= other_radius:  # label's circle is inside other_label's circle
                        containment_relations.setdefault(other_label, []).append(label)

        # Merge contained GBs into their enclosers and remove the original contained GBs
        new_merged_gbs = {}
        for encloser, contained in containment_relations.items():
            # Initialize the list for the encloser if not already present
            if encloser not in new_merged_gbs:
                new_merged_gbs[encloser] = merged_gbs[encloser][:]  # Use a copy of the original list
            # Extend the encloser's list with the contents of each contained GB
            for label in contained:
                new_merged_gbs[encloser].extend(merged_gbs[label])

        # Add GBs that are not contained by any other GB
        non_contained_labels = set(merged_gbs) - set(sum(containment_relations.values(), []))
        for label in non_contained_labels:
            if label not in new_merged_gbs:
                new_merged_gbs[label] = merged_gbs[label][:]  # Use a copy of the original list
        return new_merged_gbs

    def merge_gbs(self):
        '''
        Combining overlapping or contained balls
        '''
        gbs_sum = len(self.gbs_list)
        gbs_center, gbs_radius = [], []
        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            gbs_center.append(center)
            gbs_radius.append(radius)
        gbs_center = np.array(gbs_center)
        unvisited = [i for i in range(gbs_sum)]
        cluster = [-1 for _ in range(gbs_sum)]
        k = -1

        while len(unvisited) > 0:
            p = unvisited[0]
            unvisited.remove(p)
            neighbors = []
            for i in range(gbs_sum):
                if i != p:
                    dis = np.linalg.norm(gbs_center[i] - gbs_center[p])
                    if dis <= (gbs_radius[i] + gbs_radius[p]):
                        neighbors.append(i)
            k += 1
            cluster[p] = k
            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)
                    neighbors_pi = []
                    for j in range(gbs_sum):
                        if j != pi:
                            dis_pi = np.linalg.norm(gbs_center[j] - gbs_center[pi])
                            if dis_pi <= (gbs_radius[j] + gbs_radius[pi]):
                                neighbors_pi.append(j)
                    for t in neighbors_pi:
                        if t not in neighbors:
                            neighbors.append(t)
                if cluster[pi] == -1:
                    cluster[pi] = k

        merged_gbs = {}
        for idx, label in enumerate(cluster):
            if label not in merged_gbs:
                merged_gbs[label] = self.gbs_list[idx]
            else:
                merged_gbs[label].extend(self.gbs_list[idx])

        new_merged_gbs = self.merge_contained_gbs(merged_gbs)
        self.gbs_list = list(new_merged_gbs.values())
        return self.gbs_list

    def run(self, merge_single_gb=1, merge=1):
        '''
        Performs the ball generation and merging process and finally draws the ball
        :param merge_single_gb: 1: combined single point ball; 0: not combined
        :param merge: 1: Combining overlapping or mutually contained balls; 0: not combined
        :return:
        '''
        self.generate_gbs()
        if merge_single_gb == 1:
            self.assign_single_point_to_gbs()

        if merge == 1:
            self.merge_gbs()

        for gb in self.gbs_list:
            center, radius = self._gb_center_radius(gb)
            self.gbs_center.append(center)
            self.gbs_radius.append(radius)
        self.plot_gbs()
        return self.gbs_list, self.gbs_center, self.gbs_radius

    def plot_gbs(self, plt_type=0):
        plt.figure()
        plt.axis()
        for i in range(len(self.gbs_list)):
            # Plot points in each GB as black dots
            if plt_type == 0:
                for j in range(len(self.gbs_list[i])):
                    point_index = self.gbs_list[i][j]
                    x = self.data.loc[point_index, 'x']
                    y = self.data.loc[point_index, 'y']
                    plt.plot(x, y, '.', c='k')
                    # Add index label next to each point
                    plt.text(x, y, str(point_index), color='black', fontsize=8, ha='left')

            # Plot circles and their centers
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = self.gbs_center[i][0] + self.gbs_radius[i] * np.cos(theta)
                y = self.gbs_center[i][1] + self.gbs_radius[i] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)
                center_marker = 'x' if plt_type == 0 else '.'
                plt.plot(self.gbs_center[i][0], self.gbs_center[i][1], center_marker, color='r')
                # Add index label near the circle center
                plt.text(self.gbs_center[i][0], self.gbs_center[i][1], str(i), color='blue', fontsize=12, ha='right')

        plt.show()

# Generate balls in a cluster
def generate_gbs(dataframe, clusters_list, k, merge=1, merge_single_gb=1):
    gbs_list = {i: [] for i in range(len(clusters_list))}
    gbs_center = {i: [] for i in range(len(clusters_list))}
    for i, cluster in clusters_list.items():
        print('********** Cluster', i, '**********')
        cluster_data = dataframe.loc[list(cluster)]

        gbs_generator = GenerateGBs(cluster_data, k)
        gbs, centers, gbs_radius = gbs_generator.run(merge_single_gb, merge)

        gbs_list[i] = gbs
        gbs_center[i] = centers
    return gbs_list, gbs_center



# ******************************  Overall planning, treating each gbs as a point, sorting with ACA  *******************************


class ACA_GB:
    def __init__(self, depot_location, gbs_center, func, n_dim,
                 distance_matrix=None,
                 size_pop=10, max_iter=20,
                 alpha=1, beta=2, rho=0.1, epsilon=0.1):
        self.func = func
        self.nodes_sum = n_dim  # Number of customers
        self.ants_sum = size_pop  # Number of ants
        self.max_iter = max_iter  # Number of iterations
        self.alpha = alpha  # Importance of pheromones
        self.beta = beta  # Importance of adaptation
        self.rho = rho  # Pheromone evaporation rate
        self.epsilon = epsilon  # The pheromone factor in elite ants
        self.matrix_distance = distance_matrix
        self.matrix_heuristic = 1 / (self.matrix_distance + 1e-10 * np.eye(n_dim, n_dim))  # Avoiding divide-by-zero errors

        self.matrix_pheromone = np.ones((n_dim, n_dim))  # pheromone matrix
        self.ants_route = np.zeros((size_pop, n_dim)).astype(int)  # The path of each ant in a given generation

        self.ants_cost = None  # Total distance by each ant in a given generation
        self.elite_ant_history, self.elite_ant_cost_history = [], []  # Recording the best of each generation
        self.best_x, self.best_y = None, None # final path, final cost
        self.depot_location = depot_location  # Indexed List of Depots
        self.gbs_center = gbs_center

    def compute_transition_probabilities(self, current_node, unvisited):
        tau = self.matrix_pheromone[current_node, unvisited]
        eta = self.matrix_heuristic[current_node, unvisited]
        prob = (tau ** self.alpha) * (eta ** self.beta)
        prob /= prob.sum()
        return prob

    def update_pheromone(self, elite_ant, elite_ant_cost, ants_cost):
        # Computing Ant Pheromones
        delta_tau = np.zeros((self.nodes_sum, self.nodes_sum))
        for j in range(self.ants_sum):  # Each ant
            for k in range(self.nodes_sum - 1):  # each node
                n1, n2 = self.ants_route[j, k], self.ants_route[j, k + 1]  # Ants crawling from node n1 to node n2
                delta_tau[n1, n2] += 1 / ants_cost[j]
            n1, n2 = self.ants_route[j, self.nodes_sum - 1], self.ants_route[j, 0]
            delta_tau[n1, n2] += 1 / ants_cost[j]

        # Computing pheromones in elite ants
        delta_tau_elite = np.zeros((self.nodes_sum, self.nodes_sum))
        for k in range(self.nodes_sum - 1):
            n1, n2 = elite_ant[k], elite_ant[k + 1]
            delta_tau_elite[n1, n2] += 1 / elite_ant_cost
        n1, n2 = elite_ant[self.nodes_sum - 1], elite_ant[0]
        delta_tau_elite[n1, n2] += 1 / elite_ant_cost

        self.matrix_pheromone = (1 - self.rho) * self.matrix_pheromone + delta_tau + self.epsilon * delta_tau_elite

    def compute_total_cost_with_depots(self, route, distance_matrix):
        '''
        This method calculates the total cost of a given route, including the distances
        from the start and end points of the route to the nearest depots.
        '''
        # Extract the coordinates of the route nodes
        route_coordinates = self.gbs_center[route]

        # Calculate the distance from route[0] to each depot
        start_distances = np.linalg.norm(np.array(self.depot_location) - route_coordinates[0], axis=1)
        # Calculate the distance from route[-1] to each depot
        end_distances = np.linalg.norm(np.array(self.depot_location) - route_coordinates[-1], axis=1)

        # Find the minimum distance
        min_start_distance = start_distances.min()
        min_end_distance = end_distances.min()

        # Calculate the cost of the route
        route_cost = self.func(route, distance_matrix)

        # The total cost includes the additional distances from the start and end to the nearest depots
        total_cost = route_cost + min_start_distance + min_end_distance

        return total_cost

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for _ in range(self.max_iter):  # For each iteration
            for j in range(self.ants_sum):  # For each ant
                self.ants_route[j, 0] = np.random.randint(0, self.nodes_sum)  # Randomly select the starting point
                for k in range(self.nodes_sum - 1):  # For each node the ant visits
                    taboo_set = set(
                        self.ants_route[j, :k + 1])  # Nodes already visited and the current node, cannot be revisited
                    unvisited = list(set(range(self.nodes_sum)) - taboo_set)  # Choose from these nodes

                    # Select the next node
                    prob = self.compute_transition_probabilities(self.ants_route[j, k], unvisited)
                    if np.random.rand() <= 0.5:  # 0.5 is an assumed value for rand_0, can be adjusted
                        next_point = unvisited[np.argmax(prob)]
                    else:
                        next_point = np.random.choice(unvisited, size=1, p=prob)[0]
                    self.ants_route[j, k + 1] = next_point

            # Calculate the cost
            ants_cost = np.array([self.func(i, self.matrix_distance) for i in self.ants_route])
            # ants_cost = np.array(
            #     [self.compute_total_cost_with_depots(i, self.matrix_distance) for i in self.ants_route])

            # Record the best situation in history
            index_elite_ant = ants_cost.argmin()
            elite_ant, elite_ant_cost = self.ants_route[index_elite_ant, :].copy(), ants_cost[index_elite_ant].copy()
            self.elite_ant_history.append(elite_ant)
            self.elite_ant_cost_history.append(elite_ant_cost)

            # Update pheromones
            self.update_pheromone(elite_ant, elite_ant_cost, ants_cost)

        best_generation = np.array(self.elite_ant_cost_history).argmin()
        self.best_x = self.elite_ant_history[best_generation]
        self.best_y = self.elite_ant_cost_history[best_generation]
        print(self.best_y)
        return self.best_x, self.best_y

    fit = run



# ******************************  Partial planning, ACA planning for points within each ball  *******************************

def add_gb_centers_in_dataframe(df, gb_centers):
    # Convert gb_centers to DataFrame
    all_centers = []
    for key, centers in gb_centers.items():
        for center in centers:
            all_centers.append({'x': center[0], 'y': center[1], 'cluster': key})

    new_coords_df = pd.DataFrame(all_centers)
    new_coords_df['demand'] = 0
    new_coords_df['ReadyTime'] = 0
    new_coords_df['DueDate'] = 0
    new_coords_df['ServiceTime'] = 0
    new_coords_df['Type'] = 'new'

    # Append to the existing DataFrame
    df = pd.concat([df, new_coords_df], ignore_index=True)

    # Return the indices of gb_centers in the new DataFrame
    gb_centers_indices = {}
    for key, centers in gb_centers.items():
        indices = []
        for center in centers:
            index = df[(df['x'] == center[0]) & (df['y'] == center[1])].index[0]
            indices.append(index)
        gb_centers_indices[key] = indices

    return df, gb_centers_indices


def plot_points(df, final_route):
    fig, ax = plt.subplots(figsize=(10, 10))

    customer_index = df[df['Type'] == 'c'].index.tolist()
    depot_index = df[df['Type'] == 'd'].index.tolist()
    cs_index = df[df['Type'] == 'f'].index.tolist()

    # Plot different types of points
    ax.scatter(df.loc[customer_index, 'x'], df.loc[customer_index, 'y'], c='blue', marker='o', s=25, label='Customer')
    ax.scatter(df.loc[depot_index, 'x'], df.loc[depot_index, 'y'], c='red', marker='s', s=50, label='Depot')
    ax.scatter(df.loc[cs_index, 'x'], df.loc[cs_index, 'y'], c='green', marker='^', s=55, label='CS')

    # Label each point with its index
    for idx in customer_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in depot_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in cs_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')

    # Plot the path
    path_coords = df.loc[final_route, ['x', 'y']]
    ax.plot(path_coords['x'], path_coords['y'], c='purple', marker='o', linestyle='-', linewidth=2, markersize=5, label='Path')

    # Add legend
    ax.legend()

    # Set title and axis labels
    ax.set_title('Locations of Customers, Depots, and CS')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    plt.show()


def plot_final_routes(df, final_route):
    fig, ax = plt.subplots(figsize=(10, 10))

    customer_index = df[df['Type'] == 'c'].index.tolist()
    depot_index = df[df['Type'] == 'd'].index.tolist()
    cs_index = df[df['Type'] == 'f'].index.tolist()

    # Plot different types of points
    ax.scatter(df.loc[customer_index, 'x'], df.loc[customer_index, 'y'], c='blue', marker='o', s=25, label='Customer')
    ax.scatter(df.loc[depot_index, 'x'], df.loc[depot_index, 'y'], c='red', marker='s', s=50, label='Depot')
    ax.scatter(df.loc[cs_index, 'x'], df.loc[cs_index, 'y'], c='green', marker='^', s=55, label='CS')

    # Label each point with its index
    for idx in customer_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in depot_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in cs_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')

    # Plot each path
    for route in final_route:
        path_coords = df.loc[route, ['x', 'y']]
        ax.plot(path_coords['x'], path_coords['y'], marker='o', linestyle='-', linewidth=2, markersize=5)

    # Add legend
    ax.legend()

    # Set title and axis labels
    ax.set_title('Locations of Customers, Depots, and CS')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    plt.show()


def plot_iter_and_route(df, route, elite_ant_cost_history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the best path
    route = np.array(route)
    best_points_coordinate = df.loc[route, ['x', 'y']].values
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r', label="Best Path")
    ax[0].set_title("Best Path")
    ax[0].set_xlabel("X coordinate")
    ax[0].set_ylabel("Y coordinate")
    ax[0].legend()

    # Plot the history of the best path cost
    pd.DataFrame(elite_ant_cost_history).cummin().plot(ax=ax[1])
    ax[1].set_title("Elite Ant Cost History")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Cost")

    plt.tight_layout()
    plt.show()


def total_distance_matrix(dataframe):
    coordinates = dataframe[['x', 'y']].values
    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
    return distance_matrix


def cal_total_distance(routine, distance_matrix):
    num_points = len(routine)
    return sum(distance_matrix[routine[i], routine[i + 1]] for i in range(num_points - 1))


def calculate_energy_consumption(u_ijk, v_ijk_R, t_ijk_R, phi_motor=1.184692, phi_battery=1.112434, g=9.8,
                                 theta_ij=0, C_r=0.012, L=3000, R=0.7, A=3.8, rho=1.2041):
    # The rolling and air resistance components of energy consumption
    resistance_energy = (g * np.sin(theta_ij) + C_r * g * np.cos(theta_ij)) * (L + u_ijk) / 3600
    # The aerodynamic drag component of energy consumption
    aerodynamic_drag_energy = (R * A * rho * v_ijk_R ** 2) / 76140
    # Total energy consumption for the EV
    e_ijk_R = phi_motor * phi_battery * (resistance_energy + aerodynamic_drag_energy) * v_ijk_R * t_ijk_R
    return e_ijk_R


def calculate_charging_time(q_ik):
    charging_time = q_ik / (0.9 * 60)
    return charging_time


def calculate_one_node_energy_time(distance, departure_time, u_ijk):
    # print('distance, departure_time, u_ijk', distance, departure_time, u_ijk)
    max_periods = 24  # Maximum number of time periods
    period_length = 1  # Time period length
    travel_speed = 0
    total_energy = 0
    travel_time = 0
    remaining_distance = distance

    while remaining_distance > 0:
        # Converting time to 24-hour format
        current_time = departure_time % 24
        period = int(current_time)
        period_start = period * period_length
        period_end = (period + 1) * period_length


        # Set the speed according to the time period
        if (0 <= current_time <= 2) or (10 <= current_time <= 12):
            travel_speed = 30
        else:
            travel_speed = 60
        # print(current_time, period, period_end, travel_speed)
        # Calculates the travel time in the current time period
        t_ijk_R = min(period_end - current_time, remaining_distance / travel_speed)
        # print(t_ijk_R)

        # Calculate the energy consumption in the current time period
        energy_this_period = calculate_energy_consumption(u_ijk, travel_speed, t_ijk_R)
        # print(energy_this_period)

        # Update total energy consumption and total travel time
        total_energy += energy_this_period
        travel_time += t_ijk_R

        # Update remaining distance
        remaining_distance -= travel_speed * t_ijk_R

        # Update departure time
        departure_time += t_ijk_R

    return total_energy, travel_time, travel_speed


def remaining_route_energy(df, route, prev_i, departure_time1, load, dist_matrix):
    need_charging_energy = 0
    j = prev_i + 2
    while j < len(route):
        dis1 = dist_matrix[route[j - 1], route[j]]
        energy1, t_ijk1, speed1 = calculate_one_node_energy_time(dis1, departure_time1, load)

        service_time_minutes1 = df.loc[route[j]]['ServiceTime']
        service_time1 = service_time_minutes1 / 60
        # print('load', load, demands_list[self.route[j]])
        departure_time1 += t_ijk1 + service_time1
        load -= df.loc[route[j]]['demand']
        need_charging_energy += energy1

        j += 1
    return need_charging_energy


def total_cost(df, route, travel_time, charging_time, service_time,
               c1=120, c2=0.5, c3=0.3, c4=0.6):
    if df.loc[route[0], 'Type'] == 'd':
        dispatch_cost = c1 * 1
    else:
        dispatch_cost = 0
    travel_cost = c2 * 60 * travel_time
    service_cost = c3 * 60 * service_time
    charging_cost = c4 * 60 * charging_time
    cost = dispatch_cost + travel_cost + service_cost + charging_cost
    return cost, (dispatch_cost, travel_cost, service_cost, charging_cost)


def insert_cs_before_gb(df, dist_matrix, new_route, Q, load, departure_time, Q_max=40):
    insert_cs_in_start = False
    total_travel_time, total_charging_time, total_service_time = 0, 0, 0
    total_dist = 0

    # Find a location where you can plug in a CS
    cs_list = df[df['Type'] == 'f'].index.tolist()
    to_cs_dist = [dist_matrix[new_route[0]][j] for j in cs_list]
    min_distance_index = np.argmin(to_cs_dist)
    nearest_cs_index = cs_list[min_distance_index]
    nearest_cs_distance = to_cs_dist[min_distance_index]
    to_cs_energy, to_cs_time, to_cs_speed = calculate_one_node_energy_time(nearest_cs_distance, departure_time, load)

    if Q - to_cs_energy >= 0:
        q_ik = Q_max - (Q - to_cs_energy)
        charging_time = calculate_charging_time(q_ik)
        total_charging_time += charging_time
        total_travel_time += to_cs_time

        new_route.insert(1, nearest_cs_index)
        Q = Q_max
        departure_time += to_cs_time + charging_time
        total_dist += nearest_cs_distance

        insert_cs_in_start = True
        return new_route, Q, load, departure_time, total_travel_time, total_charging_time, total_service_time, total_dist
    else:
        return None


def routing_time(df, dist_matrix, route, Q, load, departure_time, Q_max=40):
    insert_cs = False
    initial_Q, initial_load, initial_departure_time = Q, load, departure_time
    total_charging_time = 0
    total_travel_time = 0
    total_service_time = 0
    total_dist = 0

    cs_list = df[df['Type'] == 'f'].index.tolist()
    depot_list = df[df['Type'] == 'd'].index.tolist()

    print('route', route)
    states = []  # Track the status of each node
    i = 1
    while i < len(route):
        dis = dist_matrix[route[i-1], route[i]]
        energy, t_ijk, speed = calculate_one_node_energy_time(dis, departure_time, load)
        total_travel_time += t_ijk
        total_dist += dis

        service_time_minutes = df.loc[route[i]]['ServiceTime']
        service_time = service_time_minutes / 60
        total_service_time += service_time

        load -= df.loc[route[i], 'demand']

        departure_time += t_ijk + service_time
        Q -= energy

        # Determine if the point can reach the CS
        to_cs_dist = [dist_matrix[route[i]][j] for j in cs_list]
        min_distance_index = np.argmin(to_cs_dist)
        nearest_cs_index = cs_list[min_distance_index]
        nearest_cs_distance = to_cs_dist[min_distance_index]
        to_cs_energy, to_cs_time, to_cs_speed = calculate_one_node_energy_time(nearest_cs_distance, departure_time, load)

        print(route[i-1], '->', route[i], 'energy:', energy, 'remaining power:', Q)
        # print('remaing power after to_cs_energy', Q-to_cs_energy)
        if Q - to_cs_energy >= 0 or (Q - to_cs_energy < 0 and route[i] in depot_list):
            states.append((i, route[:i+1], Q, departure_time, load, nearest_cs_index, nearest_cs_distance, to_cs_energy,
                           to_cs_time, to_cs_speed, total_travel_time, total_charging_time, total_service_time, insert_cs, total_dist))
        else:
            print('***** Insert CS *****')
            insert_cs = True
            # If the status list is empty and cannot be continued,
            # it means that a depot should be inserted at the last point of the previous gb path.
            if not states:
                (new_route, Q, load, departure_time, to_cs_travel_time, to_cs_charging_time,
                 to_cs_service_time, to_cs_dist1) = insert_cs_before_gb(df, dist_matrix, route, initial_Q, initial_load, initial_departure_time)
                route = new_route
                total_travel_time += to_cs_travel_time
                total_service_time += to_cs_service_time
                total_charging_time += to_cs_charging_time
                total_dist += to_cs_dist1
                states.append((i, route[:i + 1], Q, departure_time, load, None, None, None, None, None,
                               total_travel_time, total_charging_time, total_service_time, insert_cs, total_dist))
            else:
                (prev_i, prev_route, prev_Q, prev_departure_time, prev_load, prev_nearest_cs_index, prev_nearest_cs_distance,
                 prev_to_cs_energy, prev_to_cs_time, prev_to_cs_speed,
                 prev_total_travel_time, prev_total_charging_time, prev_total_service_time, insert_cs, prev_dist) = states[-1]

                # print(states[-1])
                total_travel_time = prev_total_travel_time + prev_to_cs_time
                total_service_time = prev_total_service_time

                q_ik = Q_max - (prev_Q - prev_to_cs_energy)
                charging_time = calculate_charging_time(q_ik)
                total_charging_time += charging_time

                route.insert(prev_i + 1, prev_nearest_cs_index)
                print('inserted_cs_route', route)
                departure_time = prev_departure_time + prev_to_cs_time + charging_time
                Q = Q_max
                load = prev_load

                total_dist = prev_dist + prev_nearest_cs_distance

                states.append((i, route[:i+1], Q, departure_time, load, None, None, None, None, None,
                               total_travel_time, total_charging_time, total_service_time, insert_cs, total_dist))
                i = prev_i + 1

        i += 1
    return states


class ACA_object:
    def __init__(self, nodes_list, distance_matrix,
                 size_ants=10, max_iter=20,
                 alpha=1, beta=2, rho=0.1, epsilon=0.1):

        self.nodes_list = nodes_list
        self.nodes_sum = len(nodes_list)  # Number of customers
        self.ants_sum = size_ants  # Number of ants
        self.max_iter = max_iter  # Number of iterations
        self.alpha = alpha  # Pheromone Important Factors
        self.beta = beta  # Important Factors in Adaptation
        self.rho = rho  # Pheromone evaporation rate
        self.epsilon = epsilon  # The pheromone factor in elite ants

        self.matrix_distance = distance_matrix[np.ix_(nodes_list, nodes_list)]
        self.matrix_heuristic = 1 / (self.matrix_distance + 1e-10 * np.eye(self.nodes_sum, self.nodes_sum))  # Avoiding divide-by-zero errors

        self.matrix_pheromone = np.ones((self.nodes_sum, self.nodes_sum))
        self.ants_route = np.zeros((size_ants, self.nodes_sum)).astype(int)

        self.ants_cost = None
        self.elite_ant_history, self.elite_ant_cost_history, self.elite_ant_info_history = [], [], []  # Recording the best of each generation

        self.elite_ant_part_cost_history = []
        self.elite_ant_dist_history = []
        self.best_route, self.best_cost, self.best_info = None, None, []


    def compute_transition_probabilities(self, current_node, unvisited):
        tau = self.matrix_pheromone[current_node, unvisited]
        eta = self.matrix_heuristic[current_node, unvisited]
        prob = (tau ** self.alpha) * (eta ** self.beta)
        prob /= prob.sum()
        return prob

    def update_pheromone(self, elite_ant, elite_ant_cost, ants_cost):
        # Calculate pheromones for all ants
        delta_tau = np.zeros((self.nodes_sum, self.nodes_sum))
        for j in range(self.ants_sum):  # For each ant
            for k in range(self.nodes_sum - 1):  # For each node
                n1, n2 = self.ants_route[j, k], self.ants_route[j, k + 1]  # Ant moves from node n1 to node n2
                delta_tau[n1, n2] += 1 / ants_cost[j]  # Apply pheromone
            n1, n2 = self.ants_route[j, self.nodes_sum - 1], self.ants_route[
                j, 0]  # Ant moves from the last node back to the first node
            delta_tau[n1, n2] += 1 / ants_cost[j]  # Apply pheromone

        # Calculate pheromones for the elite ant
        delta_tau_elite = np.zeros((self.nodes_sum, self.nodes_sum))
        for k in range(self.nodes_sum - 1):
            n1, n2 = elite_ant[k], elite_ant[k + 1]
            delta_tau_elite[n1, n2] += 1 / elite_ant_cost
        n1, n2 = elite_ant[self.nodes_sum - 1], elite_ant[0]
        delta_tau_elite[n1, n2] += 1 / elite_ant_cost

        # Pheromone evaporation and application
        self.matrix_pheromone = (1 - self.rho) * self.matrix_pheromone + delta_tau + self.epsilon * delta_tau_elite

    def insert_depot(self, df_nodes_idx, para_depot):
        # Insert depots at the beginning and end of routes
        depot_list = df[df['Type'] == 'd'].index.tolist()
        depot_coords = df.loc[depot_list, ['x', 'y']].values.astype('float')
        depots = []
        for idx, ant_route in enumerate(self.ants_route):
            if para_depot == 1:
                customer_coords = df.loc[df_nodes_idx[ant_route[0]], ['x', 'y']].values.astype('float')
            elif para_depot == 2:
                customer_coords = df.loc[df_nodes_idx[ant_route[-1]], ['x', 'y']].values.astype('float')
            elif para_depot == 3:
                customer_coords_start = df.loc[df_nodes_idx[ant_route[0]], ['x', 'y']].values.astype('float')
                customer_coords_end = df.loc[df_nodes_idx[ant_route[-1]], ['x', 'y']].values.astype('float')
                distances_start = np.linalg.norm(depot_coords - customer_coords_start, axis=1)
                distances_end = np.linalg.norm(depot_coords - customer_coords_end, axis=1)
                nearest_depot_index_start = depot_list[np.argmin(distances_start)]
                nearest_depot_index_end = depot_list[np.argmin(distances_end)]
                depots.append((nearest_depot_index_start, nearest_depot_index_end))
                continue
            else:
                continue

            distances = np.linalg.norm(depot_coords - customer_coords, axis=1)
            nearest_depot_index = depot_list[np.argmin(distances)]
            depots.append(nearest_depot_index)
        return depots

    def run(self, df, dist_matrix, Q, load, departure_time, next_gb, gb_center_idx, para_depot, max_iter=None):
        """
        :param para_depot: 0: no depot inserted; 1: initial depot inserted; 2: final depot inserted; 3: initial and final depots inserted
        :param max_iter:
        :return:
        """
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            print('@@@@@@@@@@@@@@@@@@@@ Iteration', i, '@@@@@@@@@@@@@@@@@@@@@@@@@@')
            for j in range(self.ants_sum):
                if para_depot == 1 or para_depot == 3:
                    self.ants_route[j, 0] = np.random.randint(0, self.nodes_sum)  # Random selection of starting points
                else:
                    self.ants_route[j, 0] = 0

                for k in range(self.nodes_sum - 1):  # Each node reached by ants
                    taboo_set = set(self.ants_route[j, :k + 1])
                    unvisited = list(set(range(self.nodes_sum)) - taboo_set)

                    # Select the next node
                    rand_0 = 0.3
                    rand = np.random.rand()
                    prob = self.compute_transition_probabilities(self.ants_route[j, k], unvisited)
                    if rand <= rand_0:
                        next_point = unvisited[np.argmax(prob)]
                    else:
                        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
                        if prob.sum() == 0:
                            prob = np.ones_like(prob) / len(prob)
                        else:
                            prob /= prob.sum()
                        next_point = np.random.choice(unvisited, size=1, p=prob)[0]

                    self.ants_route[j, k + 1] = next_point


            # Calculate cost
            ants_cost, ants_part_cost = [], []
            Q_load_departure = []
            route_list = []
            ants_dist = []

            depots = self.insert_depot(self.nodes_list, para_depot)
            for idx, ant_route in enumerate(self.ants_route):
                print('------ ant', idx, '----------')
                ant_route = [self.nodes_list[i] for i in ant_route]
                # Determine if this ball needs a depot inserted at the beginning and end.
                # To connect the gb's to each other, so for each gb:
                # The first node needs to insert the last node of the previous gb
                # The last node needs to connect the center of the next gb
                if para_depot == 1:
                    ant_route.insert(0, depots[idx])
                    # Use the closest point in the next granular ball as the last node
                    closest_next_gb_point = min(next_gb, key=lambda x: dist_matrix[gb_center_idx][x])
                    ant_route.append(closest_next_gb_point)
                elif para_depot == 2:
                    ant_route.append(depots[idx])
                elif para_depot == 3:
                    ant_route.insert(0, depots[idx][0])
                    ant_route.append(depots[idx][1])
                else:
                    closest_next_gb_point = min(next_gb, key=lambda x: dist_matrix[gb_center_idx][x])
                    ant_route.append(closest_next_gb_point)

                # The result_states record the state after each node arrival
                result_states = routing_time(df, dist_matrix, ant_route, Q, load, departure_time)

                # This is broken into two parts by result_states,
                # part 1: result_states[-1] is the state of the last node (the center of the next gb),
                #         which is only useful for updating the pheromone with the cost.
                # Info used only for updating pheromones ends with "_phero"
                (_, real_route, real_Q, real_departure_time, real_load, _, _, _, _, _, real_travel_time,
                 real_charging_time, real_service_time, insert_cs, real_dist) = result_states[-1]
                # To update the pheromones, compute the cost of the combined next gb-centered paths
                cost, part_cost = total_cost(df, real_route, real_travel_time, real_charging_time, real_service_time)
                ants_cost.append(cost)
                ants_part_cost.append(part_cost)
                Q_load_departure.append([real_Q, real_load, real_departure_time])
                route_list.append(real_route)
                ants_dist.append(real_dist)

            # Record the best history and update the pheromone
            index_elite_ant = np.array(ants_cost).argmin()
            elite_ant, elite_ant_cost = self.ants_route[index_elite_ant].copy(), ants_cost[index_elite_ant].copy()

            # Updating pheromones
            self.update_pheromone(elite_ant, elite_ant_cost, ants_cost)

            elite_ant_route = route_list[index_elite_ant]
            elite_ant_part_cost = ants_part_cost[index_elite_ant]
            elite_ant_info = Q_load_departure[index_elite_ant]
            elite_ant_dist = ants_dist[index_elite_ant]


            self.elite_ant_history.append(elite_ant_route)
            self.elite_ant_cost_history.append(elite_ant_cost)
            self.elite_ant_part_cost_history.append(elite_ant_part_cost)
            self.elite_ant_info_history.append(elite_ant_info)
            self.elite_ant_dist_history.append(elite_ant_dist)

        best_generation = np.array(self.elite_ant_cost_history).argmin()
        self.best_route = self.elite_ant_history[best_generation]
        self.best_cost = (self.elite_ant_cost_history[best_generation], self.elite_ant_part_cost_history[best_generation])
        self.best_info = self.elite_ant_info_history[best_generation]
        best_dist = self.elite_ant_dist_history[best_generation]

        return self.best_route, self.best_cost, self.best_info, best_dist

    fit = run


def gb_routing(df, gbs_list, total_dist_matrix, gbs_center_idx, size_ants, max_iter, Q_max=40):
    final_route, final_cost_list, final_part_cost_list = [], [], []
    final_Q = []
    final_dist = []

    # Initialize Parameters
    Q = Q_max
    departure_time = 0
    nodes_list = [item for sublist in gbs_list for item in sublist]
    load = sum(df.loc[nodes_list, 'demand'])

    # Plan for each GB sequentially
    idx = 0
    while idx < len(gbs_list):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        current_gb = gbs_list[idx]
        # To connect the gb's to each other, so for each gb:
        # The first node needs to insert the last node of the previous gb
        # The last node needs to connect the center of the next gb.
        if idx == 0 and len(gbs_list) > 1:
            para_depot = 1  # para_depot = 1: The first GB needs to insert a depot
            gb = current_gb
            gb_center = gbs_center_idx[idx]
            next_gb = gbs_list[idx + 1]
        elif idx == 0 and len(gbs_list) == 1:
            # para_depot = 3: Show that there is only one gb in the cluster,
            # Depot needs to be inserted at the beginning and end
            para_depot = 3
            gb = current_gb
            gb_center = gbs_center_idx[idx]
            next_gb = None
        elif idx == len(gbs_list) - 1:
            para_depot = 2  # para_depot = 2: The last GB needs to insert a depot
            gb = final_route[-1:] + [x for x in current_gb if x != final_route[-1]]    # remove the node duplicated with the first node
            gb_center = gbs_center_idx[idx]
            next_gb = None
        else:
            para_depot = 0
            gb = final_route[-1:] + [x for x in current_gb if x != final_route[-1]]
            gb_center = gbs_center_idx[idx]
            next_gb = gbs_list[idx + 1]
        # print('current_gb', current_gb, next_gb)

        # print('************ GB', idx, gb)
        aca_object = ACA_object(gb, total_dist_matrix, size_ants, max_iter)

        gb_route, gb_cost, next_gb_info, route_dist = aca_object.run(df, total_dist_matrix, Q, load, departure_time, next_gb, gb_center, para_depot)
        gb_total_cost, gb_part_cost = gb_cost
        Q, load, departure_time = next_gb_info

        final_cost_list.append(gb_total_cost)
        final_part_cost_list.append(gb_part_cost)
        final_Q.append(Q)
        final_dist.append(route_dist)

        # Since the last node of the previous gb was inserted into gb,
        # the first node of gb now does not need to have a path written at the end of it
        if para_depot == 1 or para_depot == 3:
            final_route.extend(gb_route[:])
        elif para_depot == 0:
            final_route.extend(gb_route[1:])
        else:
            final_route.extend(gb_route[1:])

        idx += 1
        # # Plot the path updated after each GB, plot once for each GB
        # plot_points(df, final_route)
        # plot_route(df, final_route, aca_object.elite_ant_cost_history)

    final_cost = sum(final_cost_list)
    final_part_cost = [sum(column) for column in zip(*final_part_cost_list)]
    return final_route, final_cost, final_part_cost, final_Q[-1], final_dist


# Verify that the results are correct by calculating the total path distance directly from the distance matrix
def get_routes_distances(final_routes, total_dist_matrix):
    distances = []

    for route in final_routes:
        total_distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            total_distance += total_dist_matrix[from_node][to_node]
        distances.append(total_distance)

    return distances




if __name__ == "__main__":
    # # C101
    # df = pd.read_csv(r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\c101_21_service10.txt",
    #     sep=r'\s+')
    # # # Currently the best performing clusters in ACA
    # # # k = 1, merge = 1
    # clusters = {
    #     0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 111, 112, 113,
    #         114, 115, 116, 117, 118, 119, 120, 121},
    #     1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #         68, 69, 70, 71, 72, 73},
    #     2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100,
    #         101, 102, 103, 104, 105, 107, 108}}


    # # R101
    # df = pd.read_csv(
    #     r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\r101_21.txt',
    #     sep=r'\s+')
    # # # # Currently the best performing clusters in ACA
    # # # # k = 0.4, merge = 0
    # clusters = {
    #     0: {22, 24, 30, 31, 32, 40, 41, 51, 52, 53, 54, 55, 56, 70, 71, 72, 83, 84, 85, 86, 87, 90, 91, 92, 97, 98,
    #         99, 100, 102, 109, 111},
    #     1: {64,26, 27, 28, 29, 34, 35, 37, 38, 39, 57, 58, 59, 63, 65, 66, 67, 68, 69, 73, 80, 81, 82, 103, 104, 105, 106,
    #         107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
    #     2: {48, 23, 25, 33, 36,  42, 43, 44, 45, 46, 47, 49, 50, 60, 61, 62, 74, 75, 76, 77, 78, 79, 88, 89, 93, 94, 95,
    #         96, 101}}


    # # RC101
    # df = pd.read_csv(
    #     r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\rc101_21.txt",
    #     sep=r'\s+')
    # # # Currently the best performing clusters in ACA
    # # # k = 0.6, merge = 1
    # clusters = {
    #     0: {47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 71, 75, 82, 89, 83, 88, 92, 93, 101,
    #         102, 112, 113, 114, 115, 116, 117},
    #     1: {39, 40, 41, 42, 43, 44, 45, 46, 69, 70, 72, 73, 77, 78, 79, 80, 84, 85, 86, 87, 95, 96, 97, 98, 104, 105,
    #         106, 107, 108, 110, 118, 120},
    #     2: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 66, 67, 68, 74, 76, 81, 90, 91, 94,
    #         99, 100, 103, 109, 111, 119, 121}}


    # # C201
    # df = pd.read_csv(
    #     r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\c201_21.txt",
    #     sep=r'\s+')
    # # # Currently the best performing clusters in ACA
    # # # k = 0.8, merge = 1
    # clusters = {
    #     0: {27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    #         55, 56, 57, 58, 59, 60},
    #     1: {22, 23, 24, 25, 26, 28, 91, 92, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    #         111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
    #     2: {61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    #         88, 89, 90, 93, 95}}


    # # R201
    # df = pd.read_csv(
    #     r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\r201_21.txt",
    #     sep=r'\s+')
    # # # Currently the best performing clusters in ACA
    # # # k = 0.4, merge = 0
    # clusters = {
    #     0: {22, 24, 30, 31, 32, 40, 41, 48, 51, 52, 53, 54, 55, 56, 70, 71, 72, 83, 84, 85, 86, 87, 90, 91, 92, 97, 98, 99, 100, 102, 109, 111},
    #     1: {26, 27, 28, 29, 34, 35, 37, 38, 39, 57, 58, 59, 63,  65, 66, 67, 68, 69, 73, 80, 81, 82, 103, 104, 105, 106, 107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
    #     2: {23, 25, 33, 36, 42, 43, 44, 45, 46, 47, 49, 50, 60, 61, 62, 64,74, 75, 76, 77, 78, 79, 88, 89, 93, 94, 95, 96, 101}}


    # RC201
    # df = pd.read_csv(
    #     r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\rc201_21.txt",
    #     sep=r'\s+')
    # # Currently the best performing clusters in ACA
    # # k=0.4, merge=1
    # clusters = {
    #     0: {22, 23, 24, 25, 26, 27, 28, 29, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 75, 76, 82, 89, 91, 93, 102,
    #         117, 121},
    #     1: {30, 31, 32, 33, 34, 35, 36, 37, 38, 68, 73, 74, 78, 79, 80, 81, 86, 90, 94, 95, 96, 98, 99, 100, 103, 107,
    #         108, 109, 111, 118, 119, 120},
    #     2: {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 69, 70, 71, 72, 77, 83, 84, 85, 87, 88,
    #         92, 97, 101, 104, 105, 106, 110, 112, 113, 114, 115, 116}}

    # RC202
    df = pd.read_csv(
        r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\rc202_21.txt",
        sep=r'\s+')
    # Currently the best performing clusters in ACA
    # k=0.4, merge=1
    clusters = {
        0: {22, 23, 24, 25, 26, 27, 28, 29, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 75, 76, 82, 89, 91, 93, 102,
            117, 121},
        1: {30, 31, 32, 33, 34, 35, 36, 37, 38, 68, 73, 74, 78, 79, 80, 81, 86, 90, 94, 95, 96, 98, 99, 100, 103, 107,
            108, 109, 111, 118, 119, 120},
        2: {39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 69, 70, 71, 72, 77, 83, 84, 85, 87, 88,
            92, 97, 101, 104, 105, 106, 110, 112, 113, 114, 115, 116}}



    for i in ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    customers = df[df['Type'] == 'c']
    depot = df[df['Type'] == 'd']
    charge_station = df[df['Type'] == 'f']

    # Calculate the total demand for each cluster
    total_demand_per_cluster = {}
    for cluster_id, indices in clusters.items():
        total_demand = customers.loc[list(indices), 'demand'].sum()
        total_demand_per_cluster[cluster_id] = total_demand
    print("Total demand per cluster:", total_demand_per_cluster)

    W = 650
    Q_max = 40

    # 记录运行时间
    start_time = time.time()

    # parameter setting
    k = 0.4
    merge = 1
    gbs_list, gbs_center_location = generate_gbs(df, clusters, k, merge, merge_single_gb=1)
    # print(gbs_list, gbs_center_location)

    # Add gb_centers to df
    df, gb_centers_df_idx = add_gb_centers_in_dataframe(df, gbs_center_location)
    total_dist_matrix = total_distance_matrix(df)


    routes = []
    cost_cluster, part_cost_cluster = [], []
    remaining_Q = []
    dist1 = []
    for i, cluster in list(gbs_list.items())[:1]:

        # Planning granular ball paths
        centers1 = np.array(gbs_center_location[i])
        depot_loction = np.array(depot[['x', 'y']])
        gbs_distance_matrix = cdist(centers1, centers1, metric='euclidean')
        aca_gb = ACA_GB(depot_loction, centers1, func=cal_total_distance, n_dim=len(centers1),
                        distance_matrix=gbs_distance_matrix,
                        size_pop=10, max_iter=50)

        gbs_order, _ = aca_gb.run()
        # Because gbs is directionless, set the 0.5 to change the path direction
        random_number = np.random.rand()
        if random_number > 0.5:
            gbs_order = gbs_order[::-1]

        gbs_list1 = [gbs_list[i][j] for j in gbs_order]
        gbs_center_idx = [gb_centers_df_idx[i][j] for j in gbs_order]
        print(gbs_order, gbs_list1)

        # Planning paths inside the granular ball
        size_ants, Iter = 3, 5
        final_route, final_cost, final_part_cost, final_Q, final_route_dist = gb_routing(df, gbs_list1, total_dist_matrix, gbs_center_idx, size_ants, Iter)
        routes.append(final_route)
        cost_cluster.append(final_cost)
        part_cost_cluster.append(final_part_cost)
        remaining_Q.append(final_Q)
        dist1.append(final_route_dist)

        plot_points(df, final_route)


    end_time = time.time()
    print(f"Running time: {end_time - start_time:.4f}s")

    routes_dist = get_routes_distances(routes, total_dist_matrix)
    part_cost_cluster_sums = [sum(x) for x in zip(*part_cost_cluster)]
    print('final_route =', routes)
    print('total_cost =', sum(cost_cluster))
    print('part_cost =', part_cost_cluster_sums)
    print('routes_dist =', sum(routes_dist), routes_dist)
    flattened_list = [item for sublist in dist1 for item in sublist]
    print('dist =',  sum(flattened_list), [sum(sublist) for sublist in dist1], dist1)
    print('remaining_Q =', sum(remaining_Q), remaining_Q)
    print('each_cluster_cost =', cost_cluster, part_cost_cluster)
    plot_final_routes(df, routes)





