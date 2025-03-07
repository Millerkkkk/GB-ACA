#  coding: UTF-8  #
"""
@Project ：GB-MDHOEVRP 
@File    ：genetic algorithm.py
@IDE     ：PyCharm 
@Author  ：Yingkai
@Date    ：2025/3/3 13:56 
"""
import copy
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix






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

        # if 0 <= current_time < 2:  # 07:00 - 09:00
        #     travel_speed = 30
        # elif 2 <= current_time < 5:  # 09:00 - 12:00
        #     travel_speed = 55
        # elif 5 <= current_time < 7:  # 12:00 - 14:00
        #     travel_speed = 50
        # elif 7 <= current_time < 10:  # 14:00 - 17:00
        #     travel_speed = 55
        # elif 10 <= current_time < 12:  # 17:00 - 19:00
        #     travel_speed = 30
        # else:
        #     # Set speed for all other times as default (e.g., off-peak hours)
        #     travel_speed = 60

        # Calculates the travel time in the current time period
        t_ijk_R = min(period_end - current_time, remaining_distance / travel_speed)

        # Calculate the energy consumption in the current time period
        energy_this_period = calculate_energy_consumption(u_ijk, travel_speed, t_ijk_R)

        # Update total energy consumption and total travel time
        total_energy += energy_this_period
        travel_time += t_ijk_R

        # Update remaining distance
        remaining_distance -= travel_speed * t_ijk_R

        # Update departure time
        departure_time += t_ijk_R

    return total_energy, travel_time, travel_speed

def calculate_remaining_route_energy(df, route, current_index, departure_time, load, dist_matrix):
    total_remaining_energy = 0
    # Iterate over the remaining route to calculate energy consumption for each segment
    for i in range(current_index, len(route) - 1):
        # Get the distance between the current node and the next node
        dis = dist_matrix[route[i], route[i + 1]]

        energy, t_ijk, speed = calculate_one_node_energy_time(dis, departure_time, load)
        total_remaining_energy += energy
        load -= df.loc[route[i + 1], 'demand']

        departure_time += t_ijk
        # Consider the service time at each node
        service_time_minutes = df.loc[route[i + 1], 'ServiceTime']
        service_time = service_time_minutes / 60
        departure_time += service_time
    return total_remaining_energy







class GA:
    def __init__(self, file_name, customer_idx, pop_size=30, cx_pb=0.85, mut_pb=0.1, n_gen=100):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, "data", file_name)
        self.customer_original_idx = customer_idx
        self.df = None
        self.customers = None
        self.depot = None
        self.CS = None
        self.depot_idx = None
        self.CS_idx = None  # CS: Charging station
        self.total_demand = None
        self.dist_matrix = None

        self.n_gen = n_gen  # Number of Generations
        self.pop_size = pop_size    # population size
        self.cx_pb = cx_pb  # Crossover Probability
        self.mut_pb = mut_pb    # Mutation Probability
        self.n_select = int(0.8* pop_size)

        self.sol_list = []
        self.fitness = []

        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.file_path, sep=r'\s+')

        for i in ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']:
            self.df[i] = pd.to_numeric(self.df[i], errors='coerce')

        self.customers = self.df.iloc[list(self.customer_original_idx)]
        self.depot = self.df[self.df['Type'] == 'd']
        self.depot_idx = self.depot.index.tolist()
        self.CS = self.df[self.df['Type'] == 'f']
        self.CS_idx = self.CS.index.tolist()
        self.total_demand = self.df.loc[list(self.customer_original_idx), 'demand'].sum()

        coords_x, coords_y = self.df['x'], self.df['y']
        coords = np.column_stack((coords_x, coords_y))
        self.dist_matrix = distance_matrix(coords, coords)


        print('customer', self.customer_original_idx)
        print('depot', self.depot_idx)
        print('CS', self.CS_idx)


    def routing_time(self, gene_route, Q_max=40):
        route = copy.deepcopy(gene_route)
        load = self.total_demand
        Q = Q_max
        departure_time = 0
        total_charging_time = 0
        total_travel_time = 0
        total_service_time = 0
        total_dist = 0

        # insert depot
        first_node = route[0]
        last_node = route[-1]
        first_depot = min(self.depot_idx, key=lambda depot: self.dist_matrix[depot, first_node])
        last_depot = min(self.depot_idx, key=lambda depot: self.dist_matrix[depot, last_node])

        route.insert(0, first_depot)
        route.append(last_depot)


        states = []  # Track the status of each node
        i = 1
        while i < len(route):
            dis = self.dist_matrix[route[i - 1], route[i]]
            energy, t_ijk, speed = calculate_one_node_energy_time(dis, departure_time, load)
            total_travel_time += t_ijk
            total_dist += dis

            service_time_minutes = self.df.loc[route[i]]['ServiceTime']
            service_time = service_time_minutes / 60
            total_service_time += service_time

            load -= self.df.loc[route[i], 'demand']

            departure_time += t_ijk + service_time
            Q -= energy

            # Determine if the point can reach the CS
            # Charging strategy: At each node, calculate the power of this node to the nearest depot,
            #                    if the current power cannot reach the nearest depot.
            #                    then backtrack to the previous node and insert the CS after the node
            to_cs_dist = [self.dist_matrix[route[i]][j] for j in self.CS_idx]
            min_distance_index = np.argmin(to_cs_dist)
            nearest_cs_index = self.CS_idx[min_distance_index]
            nearest_cs_distance = to_cs_dist[min_distance_index]
            to_cs_energy, to_cs_time, to_cs_speed = calculate_one_node_energy_time(nearest_cs_distance, departure_time,
                                                                                   load)
            # print(Q - to_cs_energy)
            if Q - to_cs_energy >= 0 or (Q - to_cs_energy < 0 and route[i] in self.depot_idx):
                states.append(
                    (i, route[:i + 1], Q, departure_time, load, nearest_cs_index, nearest_cs_distance, to_cs_energy,
                     to_cs_time, to_cs_speed, total_travel_time, total_charging_time, total_service_time,
                     total_dist))
            else:
                (prev_i, prev_route, prev_Q, prev_departure_time, prev_load, prev_nearest_cs_index,
                 prev_nearest_cs_distance,
                 prev_to_cs_energy, prev_to_cs_time, prev_to_cs_speed,
                 prev_total_travel_time, prev_total_charging_time, prev_total_service_time, prev_dist) = states[-1]

                total_travel_time = prev_total_travel_time + prev_to_cs_time
                total_service_time = prev_total_service_time

                q_ik = Q_max - (prev_Q - prev_to_cs_energy)
                charging_time = calculate_charging_time(q_ik)
                total_charging_time += charging_time

                route.insert(prev_i + 1, prev_nearest_cs_index)
                departure_time = prev_departure_time + prev_to_cs_time + charging_time
                Q = Q_max
                load = prev_load

                total_dist = prev_dist + prev_nearest_cs_distance

                states.append((i, route[:i + 1], Q, departure_time, load, None, None, None, None, None,
                               total_travel_time, total_charging_time, total_service_time, total_dist))
                i = prev_i + 1
            i += 1
        return states

    def total_cost(self, route, c1=120, c2=0.5, c3=0.3, c4=0.6):
        status = self.routing_time(route)

        (_, final_route, _, _, _, _, _, _, _, _,
         travel_time, charging_time, service_time, dist) = status[-1]

        dispatch_cost = c1
        travel_cost = c2 * 60 * travel_time
        service_cost = c3 * 60 * service_time
        charging_cost = c4 * 60 * charging_time
        cost = dispatch_cost + travel_cost + service_cost + charging_cost
        return cost, final_route

    def genInitialSol(self):
        for i in range(self.pop_size):
            shuffled_list = random.sample(list(self.customer_original_idx), len(self.customer_original_idx))
            self.sol_list.append(shuffled_list)
        return self.sol_list

    def get_fitness(self):
        self.fitness = []
        for gene in self.sol_list:
            fit, final_route = self.total_cost(gene)
            self.fitness.append(fit)
        return self.fitness

    def select(self, ps=0.5):
        population_copy = copy.deepcopy(self.sol_list)
        fitness_copy = copy.deepcopy(self.fitness)

        # Sort by adaptation
        package = list(zip(population_copy, fitness_copy))
        package.sort(key=lambda elem: elem[1])

        population_route, sorted_fitness = zip(*package)
        population_route = list(population_route)
        sorted_fitness = list(sorted_fitness)

        # Individuals selected in the top ps scale
        population_select = population_route[:int(len(population_copy) * ps)]

        # Binary Bidding Tournament Options
        while len(population_select) < len(population_copy):
            rand1, rand2 = np.random.choice(len(population_route), size=2, replace=False)

            if sorted_fitness[rand1] <= sorted_fitness[rand2]:
                population_select.append(population_route[rand1])
            else:
                population_select.append(population_route[rand2])

        self.sol_list = population_select
        self.fitness = self.get_fitness()
        return self.sol_list, self.fitness

    def cross(self, x1, x2):
        x1_copy = copy.deepcopy(x1)
        x2_copy = copy.deepcopy(x2)

        rand_number = np.random.randint(1, len(x1))

        x1_head = x1_copy[:rand_number]
        x2_head = x2_copy[:rand_number]

        k = rand_number
        while True:
            if x2_copy[k] not in x1_head:
                x1_head.append(x2_copy[k])

            if x1_copy[k] not in x2_head:
                x2_head.append(x1_copy[k])

            k += 1
            if k == len(x1_copy):
                k = 0
            if len(x1_head) == len(x1_copy) and len(x2_head) == len(x2_copy):
                break
        return x1_head, x2_head

    def crossover(self):
        population_s_copy = copy.deepcopy(self.sol_list)
        size = len(population_s_copy)

        for _ in range(int(self.cx_pb * size)):
            while True:
                rand1, rand2 = np.random.choice(len(self.sol_list), size=2, replace=False)
                if rand1 != rand2:
                    break

            progeny1, progeny2 = self.cross(self.sol_list[rand1], self.sol_list[rand2])
            self.sol_list[rand1], self.sol_list[rand2] = progeny1, progeny2

        return self.sol_list

    def mutate_indiv(self, indivv):  # inversion mutation
        indiv = copy.copy(indivv)

        def fi(x, y):  # 产生一个 x-y(不包括 y) 的随机数
            return np.random.randint(x, y)

        indiv_length = len(indiv)
        while True:
            random1 = fi(0, indiv_length)
            random2 = fi(0, indiv_length)
            if random1 != random2:
                break

        rand_min = min(random1, random2)
        rand_max = max(random1, random2)
        fanzhuan = indiv[rand_min:rand_max]
        indiv[rand_min:rand_max] = reversed(fanzhuan)  # 直接反转，不需要删除再插入
        return indiv

    def mutate_exchange(self, indivv):  # exchanged mutation
        indiv = copy.copy(indivv)

        def fi(x, y):
            return np.random.randint(x, y)

        indiv_length = len(indiv)
        while True:
            random1 = fi(0, indiv_length)
            random2 = fi(0, indiv_length)
            if random1 != random2:
                break

        indiv[random1], indiv[random2] = indiv[random2], indiv[random1]  # 直接交换
        return indiv

    def mutate_insert(self, indiv):  # insertion mutation
        indiv_copy = copy.copy(indiv)

        def fi(x, y):
            return np.random.randint(x, y)

        indiv_length = len(indiv_copy)
        while True:
            random1 = fi(0, indiv_length)
            random2 = fi(0, indiv_length)
            if random1 != random2:
                break

        rand_min = min(random1, random2)
        rand_max = max(random1, random2)

        gene = indiv_copy.pop(rand_max)  # 取出元素
        indiv_copy.insert(rand_min, gene)  # 插入到新的位置

        return indiv_copy

    def mutation(self):
        population_c_copy = copy.deepcopy(self.sol_list)
        mutate_num = int(self.mut_pb * self.pop_size)

        for _ in range(mutate_num):
            rand_num = np.random.randint(0, len(population_c_copy))
            random_float = np.random.random()
            if random_float < 0.33:
                population_c_copy[rand_num] = self.mutate_indiv(population_c_copy[rand_num])
            elif random_float < 0.66:
                population_c_copy[rand_num] = self.mutate_exchange(population_c_copy[rand_num])
            else:
                population_c_copy[rand_num] = self.mutate_insert(population_c_copy[rand_num])

        self.sol_list = population_c_copy
        return self.sol_list


def plot_final_routes(file_name, final_route):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(8, 7))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", file_name)
    df = pd.read_csv(file_path, sep=r'\s+')

    for i in ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']:
        df[i] = pd.to_numeric(df[i], errors='coerce')

    customer_index = df[df['Type'] == 'c'].index.tolist()
    depot_index = df[df['Type'] == 'd'].index.tolist()
    cs_index = df[df['Type'] == 'f'].index.tolist()

    # Plot different types of points
    ax.scatter(df.loc[customer_index, 'x'], df.loc[customer_index, 'y'], c='#3179B5', marker='o', s=45, label='Customer')
    ax.scatter(df.loc[depot_index, 'x'], df.loc[depot_index, 'y'], c='#D02C1B', marker='s', s=80, label='Depot')
    ax.scatter(df.loc[cs_index, 'x'], df.loc[cs_index, 'y'], c='#25A61F', marker='^', s=100, label='CS')

    for idx in df.index:
        ax.text(df.loc[idx, 'x'] + 0.5, df.loc[idx, 'y'] + 0.5, str(idx), fontsize=10, ha='left', va='bottom')

    # Define the specific colors for each route
    route_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green (like in the image)

    # Plot each path with a different color
    for idx, route in enumerate(final_route):
        path_coords = df.loc[route, ['x', 'y']]
        ax.plot(path_coords['x'], path_coords['y'], marker='o', linestyle='-', linewidth=1.5, markersize=3,
                color=route_colors[idx % len(route_colors)])  # Use the colormap to assign a different color to each route

    # Add legend
    ax.legend(fontsize=14)

    # # Set axis labels and title
    # plt.xlabel('X-label', fontsize=20, labelpad=10)
    # plt.ylabel('Y-label', fontsize=20, labelpad=10)
    #
    # plt.title('Test set C202', fontsize=22, pad=15)

    # Set axis ticks
    ax.set_xticks(range(0, int(df['x'].max()) + 15, 10))
    ax.set_yticks(range(0, int(df['y'].max()) + 10, 10))

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18, direction='in', top=True, right=True)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)  # Hide the first Y-axis tick label

    # Use tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()


def run_ga(file_path, clusters):
    final_route, final_cost = [], []
    total_cost = []
    total_dist = []
    for idx, cluster in clusters.items():

        # 1. Initialize GA
        customer_list = cluster
        ga_model = GA(file_path, customer_list, pop_size=60, cx_pb=0.9, mut_pb=0.4, n_gen=600)
        ga_model.genInitialSol()
        ga_model.get_fitness()
        ga_model.select(ps=0.5)
        ga_model.crossover()
        ga_model.mutation()

        # 2. set population
        population = ga_model.sol_list
        fitness = ga_model.get_fitness()
        best_idx = np.argmin(fitness)
        best_indiv = population[best_idx]
        best_indiv_cost = fitness[best_idx]
        print('best_indiv', best_indiv, best_indiv_cost)

        # 3. Record results
        best_indiv_record = [best_indiv_cost]
        population_ave_record = [sum(fitness) / len(fitness)]

        # 4. Main Loop
        for gen in range(ga_model.n_gen):
            if gen % 20 == 0:
                print(f"Current iteration {gen}, current optimal value: {best_indiv_cost}")

            # # **动态调整变异率 & 选择策略**
            # if gen < 50:
            #     ga_model.select(ps=0.4)
            # else:
            #     ga_model.select(ps=0.5)

            ga_model.crossover()
            ga_model.mutation()

            # Find the optimal individual
            population1 = ga_model.sol_list
            fitness1 = ga_model.get_fitness()
            best_idx1 = np.argmin(fitness1)
            current_best = population1[best_idx1]
            current_best_cost = fitness1[best_idx1]

            # Update
            if current_best_cost < best_indiv_cost:
                best_indiv = current_best
                best_indiv_cost = current_best_cost

            best_indiv_record.append(best_indiv_cost)
            population_ave_record.append(sum(fitness1) / len(fitness1))

        # final result
        cost, route = ga_model.total_cost(best_indiv)
        total_cost.append(cost)
        final_cost.append(best_indiv_record[-1])
        final_route.append(route)

        route_dist = sum(ga_model.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_dist.append(route_dist)


        plotObj(best_indiv_record)
        plot_final_routes(file_path, final_route)

    plot_final_routes(file_path, final_route)
    print('total_cost', sum(total_cost), total_cost)
    print('total_dist', sum(total_dist), total_dist)
    print(final_cost)
    print(final_route)





if __name__ == '__main__':
    # path = "c101_21.txt"
    #     # r"C:\Users\Millerkkk\OneDrive - Universitatea Babeş-Bolyai\Project_Python\GB-MDHOEVRP\data\c101_21.txt"
    # clusters = {
    #     0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 111, 112, 113,
    #         114, 115, 116, 117, 118, 119, 120, 121},
    #     1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #         68, 69, 70, 71, 72, 73, 97},
    #     2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,  98, 99, 100,
    #         101, 102, 103, 104, 105, 107, 108}}

    # path = "r101_21.txt"
    # clusters = {
    #         0: {22, 24, 30, 31, 32, 40, 41, 48, 51, 52, 53, 54, 55, 56, 70, 71, 72, 83, 84, 85, 86, 87, 90, 91, 92, 97, 98,
    #             99, 100, 102, 109, 111},
    #         1: {64,26, 27, 28, 29, 34, 35, 37, 38, 39, 57, 58, 59, 63, 65, 66, 67, 68, 69, 73, 80, 81, 82, 103, 104, 105, 106,
    #             107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
    #         2: {23, 25, 33, 36,  42, 43, 44, 45, 46, 47, 49, 50, 60, 61, 62, 74, 75, 76, 77, 78, 79, 88, 89, 93, 94, 95,
    #             96, 101}}

    # path = "rc101_21.txt"
    # clusters = {
    #         0: {22, 24, 26, 66, 29, 67, 25, 28, 27, 23, 91, 121, 76, 89, 82, 102, 117, 75, 93, 62, 63, 65, 64, 61, 60, 59,
    #             58, 57, 56},
    #         1: {34, 30, 31, 32, 36, 37, 38, 68, 35, 33, 99, 94, 100, 81, 109, 74, 119, 90, 111, 86, 103, 120, 73, 78, 107,
    #             95, 108, 80, 118, 96, 79, 98},
    #         2: {46, 44, 42, 69, 39, 40, 70, 45, 43, 41, 72, 97, 110, 84, 106, 105, 116, 77, 85, 104, 87, 101, 112, 113, 115,
    #             114, 92, 88, 83, 71, 55, 52, 50, 48, 47, 49, 51, 53, 55, 54}}

    # path = "c104_21.txt"
    # clusters = {
    #         0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 111, 112,
    #             113,
    #             114, 115, 116, 117, 118, 119, 120, 121},
    #         1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #             68, 69, 70, 71, 72, 73},
    #         2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100,
    #             101, 102, 103, 104, 105, 107, 108}}

    path = "r104_21.txt"
    clusters = {
            0: {22, 24, 30, 31, 32, 40, 41, 48, 51, 52, 53, 54, 55, 56, 70, 71, 72, 83, 84, 85, 86, 87, 90, 91, 92, 97, 98, 99, 100, 102, 109, 111},
            1: {26, 27, 28, 29, 34, 35, 37, 38, 39, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 73, 80, 81, 82, 103, 104, 105, 106, 107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
            2: {23, 25, 33, 36, 42, 43, 44, 45, 46, 47, 49, 50, 60, 61, 62, 74, 75, 76, 77, 78, 79, 88, 89, 93, 94, 95, 96, 101}}

    start_time = time.time()
    run_ga(path, clusters)
    end_time = time.time()
    print('run time:', end_time-start_time)