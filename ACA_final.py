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

def add_gb_centers_in_dataframe(df, gb_centers):
    # 将 gb_centers 转换为 DataFrame
    all_centers = []
    for key, centers in gb_centers.items():
        for center in centers:
            all_centers.append({'x': center[0], 'y': center[1], 'cluster': key})

    new_coords_df = pd.DataFrame(all_centers)
    new_coords_df['demand'] = 0  # 假设新坐标的需求为0
    new_coords_df['ReadyTime'] = 0  # 假设新坐标的ReadyTime为0
    new_coords_df['DueDate'] = 0  # 假设新坐标的DueDate为0
    new_coords_df['ServiceTime'] = 0  # 假设新坐标的ServiceTime为0
    new_coords_df['Type'] = 'new'  # 标记新坐标类型

    # 续到现有的 DataFrame 中
    df = pd.concat([df, new_coords_df], ignore_index=True)

    # 返回 gb_centers 在新 DataFrame 中的索引
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

    # 绘制不同类型的点
    ax.scatter(df.loc[customer_index, 'x'], df.loc[customer_index, 'y'], c='blue', marker='o', s=25, label='Customer')
    ax.scatter(df.loc[depot_index, 'x'], df.loc[depot_index, 'y'], c='red', marker='s', s=50, label='Depot')
    ax.scatter(df.loc[cs_index, 'x'], df.loc[cs_index, 'y'], c='green', marker='^', s=55, label='CS')

    # 在每个点旁边标上序号
    for idx in customer_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in depot_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in cs_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')

    # 绘制路径
    path_coords = df.loc[final_route, ['x', 'y']]
    ax.plot(path_coords['x'], path_coords['y'], c='purple', marker='o', linestyle='-', linewidth=2, markersize=5,
            label='Path')

    # 添加图例
    ax.legend()

    # 设置标题和坐标轴标签
    ax.set_title('Locations of Customers, Depots, and CS')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    plt.show()


def plot_final_routes(df, final_route):
    fig, ax = plt.subplots(figsize=(10, 10))

    customer_index = df[df['Type'] == 'c'].index.tolist()
    depot_index = df[df['Type'] == 'd'].index.tolist()
    cs_index = df[df['Type'] == 'f'].index.tolist()

    # 绘制不同类型的点
    ax.scatter(df.loc[customer_index, 'x'], df.loc[customer_index, 'y'], c='blue', marker='o', s=25, label='Customer')
    ax.scatter(df.loc[depot_index, 'x'], df.loc[depot_index, 'y'], c='red', marker='s', s=50, label='Depot')
    ax.scatter(df.loc[cs_index, 'x'], df.loc[cs_index, 'y'], c='green', marker='^', s=55, label='CS')

    # 在每个点旁边标上序号
    for idx in customer_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in depot_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')
    for idx in cs_index:
        ax.text(df.loc[idx, 'x'], df.loc[idx, 'y'], str(idx), fontsize=9, ha='right')

    # 绘制每条路径
    for route in routes:
        path_coords = df.loc[route, ['x', 'y']]
        ax.plot(path_coords['x'], path_coords['y'], marker='o', linestyle='-', linewidth=2, markersize=5)

    # 添加图例
    ax.legend()

    # 设置标题和坐标轴标签
    ax.set_title('Locations of Customers, Depots, and CS')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    plt.show()


def total_distance_matrix(dataframe):
    coordinates = dataframe[['x', 'y']].values
    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
    return distance_matrix


def cal_total_distance(routine, distance_matrix):
    num_points = len(routine)
    return sum(distance_matrix[routine[i], routine[i + 1]] for i in range(num_points - 1))


class GenerateGBs:
    def __init__(self, data, k):
        self.data = data[['x', 'y']]
        self.gbs_list = [self.data.index.tolist()]
        self.gbs_center = []
        self.gbs_radius = []
        self.threshold = k * np.sqrt(len(self.data))

    def _split_gb(self, gb_idx):
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
        while True:
            gbs_num = len(self.gbs_list)
            self.gbs_list = self.split()
            gbs_num_new = len(self.gbs_list)
            if gbs_num == gbs_num_new:
                print('break, all GBs have been generated')
                break
        return self.gbs_list

    def _gb_center_radius(self, gb):
        x_center = self.data.loc[gb, 'x'].mean()
        y_center = self.data.loc[gb, 'y'].mean()
        gb_center = (x_center, y_center)
        gb_radius = max(np.sqrt((self.data.loc[gb, 'x'] - x_center) ** 2 + (self.data.loc[gb, 'y'] - y_center) ** 2))
        return gb_center, gb_radius

    def merge_gbs(self):
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
        self.gbs_list = list(merged_gbs.values())
        return self.gbs_list

    def run(self, merge=1):
        self.generate_gbs()
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
            if plt_type == 0:
                plt.plot(self.data.loc[self.gbs_list[i], 'x'], self.data.loc[self.gbs_list[i], 'y'], '.', c='k')
            if plt_type == 0 or plt_type == 1:
                theta = np.arange(0, 2*np.pi, 0.01)
                x = self.gbs_center[i][0] + self.gbs_radius[i] * np.cos(theta)
                y = self.gbs_center[i][1] + self.gbs_radius[i] * np.sin(theta)
                plt.plot(x, y, c='r', linewidth=0.8)
                plt.plot(self.gbs_center[i][0], self.gbs_center[i][1], 'x' if plt_type == 0 else '.', color='r')
        plt.show()


class ACA_GB:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1, epsilon=0.1):
        self.func = func
        self.nodes_sum = n_dim  # 城市数量
        self.ants_sum = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度
        self.epsilon = epsilon  # 精英蚂蚁的信息素因子
        self.matrix_distance = distance_matrix
        self.matrix_heuristic = 1 / (self.matrix_distance + 1e-10 * np.eye(n_dim, n_dim))  # 避免除零错误

        self.matrix_pheromone = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新
        self.ants_route = np.zeros((size_pop, n_dim)).astype(int)  # 某一代每个蚂蚁的爬行路径

        self.ants_cost = None  # 某一代每个蚂蚁的爬行总距离
        self.elite_ant_history, self.elite_ant_cost_history = [], []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None

    def compute_transition_probabilities(self, current_node, unvisited):
        tau = self.matrix_pheromone[current_node, unvisited]
        eta = self.matrix_heuristic[current_node, unvisited]
        prob = (tau ** self.alpha) * (eta ** self.beta)
        prob /= prob.sum()
        return prob

    def update_pheromone(self, elite_ant, elite_ant_cost, ants_cost):
        # 计算蚂蚁信息素
        delta_tau = np.zeros((self.nodes_sum, self.nodes_sum))
        for j in range(self.ants_sum):  # 每个蚂蚁
            for k in range(self.nodes_sum - 1):  # 每个节点
                n1, n2 = self.ants_route[j, k], self.ants_route[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                delta_tau[n1, n2] += 1 / ants_cost[j]  # 涂抹的信息素
            n1, n2 = self.ants_route[j, self.nodes_sum - 1], self.ants_route[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
            delta_tau[n1, n2] += 1 / ants_cost[j]  # 涂抹信息素

        # 计算精英蚂蚁的信息素
        delta_tau_elite = np.zeros((self.nodes_sum, self.nodes_sum))
        for k in range(self.nodes_sum - 1):
            n1, n2 = elite_ant[k], elite_ant[k + 1]
            delta_tau_elite[n1, n2] += 1 / elite_ant_cost
        n1, n2 = elite_ant[self.nodes_sum - 1], elite_ant[0]
        delta_tau_elite[n1, n2] += 1 / elite_ant_cost

        # 信息素飘散+信息素涂抹
        self.matrix_pheromone = (1 - self.rho) * self.matrix_pheromone + delta_tau + self.epsilon * delta_tau_elite

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for _ in range(self.max_iter):  # 对每次迭代

            for j in range(self.ants_sum):  # 对每个蚂蚁
                self.ants_route[j, 0] = np.random.randint(0, self.nodes_sum)  # 随机选择起点
                for k in range(self.nodes_sum - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.ants_route[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    unvisited = list(set(range(self.nodes_sum)) - taboo_set)  # 在这些点中做选择

                    # 选择下个节点
                    prob = self.compute_transition_probabilities(self.ants_route[j, k], unvisited)
                    if np.random.rand() <= 0.5:  # 0.5是rand_0的假设值，可以调整
                        next_point = unvisited[np.argmax(prob)]
                    else:
                        next_point = np.random.choice(unvisited, size=1, p=prob)[0]
                    self.ants_route[j, k + 1] = next_point

                # print('self.ants_route', self.ants_route)


            # 计算距离
            ants_cost = np.array([self.func(i, self.matrix_distance) for i in self.ants_route])

            # 顺便记录历史最好情况
            index_elite_ant = ants_cost.argmin()
            elite_ant, elite_ant_cost = self.ants_route[index_elite_ant, :].copy(), ants_cost[index_elite_ant].copy()
            self.elite_ant_history.append(elite_ant)
            self.elite_ant_cost_history.append(elite_ant_cost)

            # 更新信息素
            self.update_pheromone(elite_ant, elite_ant_cost, ants_cost)

        best_generation = np.array(self.elite_ant_cost_history).argmin()
        self.best_x = self.elite_ant_history[best_generation]
        self.best_y = self.elite_ant_cost_history[best_generation]
        return self.best_x, self.best_y

    fit = run


def generate_gbs(dataframe, clusters_list, k):
    gbs_list = {i: [] for i in range(len(clusters_list))}
    gbs_center = {i: [] for i in range(len(clusters_list))}
    for i, cluster in clusters_list.items():
        print('**********', i, '**********')
        cluster_data = dataframe.loc[list(cluster)]

        gbs_generator = GenerateGBs(cluster_data, k)
        gbs, centers, gbs_radius = gbs_generator.run(1)
        gbs_list[i] = gbs
        gbs_center[i] = centers
    return gbs_list, gbs_center


# ******************************  整体规划，对每个GB内进行规划  *******************************
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
    # print('total cost111111111111111111111111111111111111111111111111111111', route)
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
    # 查找可以插入充电站的位置
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

        del new_route[0]  # 首个元素是上一个gb的最后一个元素，需要删除后再插入充电站
        new_route.insert(0, nearest_cs_index)
        Q = Q_max
        departure_time += to_cs_time + charging_time

        insert_cs_in_start = True
        return new_route, Q, load, departure_time, total_travel_time, total_charging_time, total_service_time
    else:
        return None


def routing_time(df, dist_matrix, route, Q, load, departure_time, Q1, load1, departure_time1, Q_max=40):
    insert_cs = False
    total_charging_time = 0
    total_travel_time = 0
    total_service_time = 0

    cs_list = df[df['Type'] == 'f'].index.tolist()

    states = []  # Track the status of each node
    i = 1
    while i < len(route):
        # print('route-Q-load-departure_time', route, Q, load, departure_time)
        dis = dist_matrix[route[i-1], route[i]]
        energy, t_ijk, speed = calculate_one_node_energy_time(dis, departure_time, load)
        total_travel_time += t_ijk

        service_time_minutes = df.loc[route[i]]['ServiceTime']
        service_time = service_time_minutes / 60
        total_service_time += service_time

        load -= df.loc[route[i], 'demand']

        departure_time += t_ijk + service_time
        Q -= energy

        # print(route[i - 1], route[i])
        # print('energy, t_ijk, speed', energy, t_ijk, speed, total_travel_time, total_service_time, departure_time)

        # 判断该点是否能到达CS
        to_cs_dist = [dist_matrix[route[i]][j] for j in cs_list]
        min_distance_index = np.argmin(to_cs_dist)
        nearest_cs_index = cs_list[min_distance_index]
        nearest_cs_distance = to_cs_dist[min_distance_index]
        to_cs_energy, to_cs_time, to_cs_speed = calculate_one_node_energy_time(nearest_cs_distance, departure_time, load)
        # print(Q - to_cs_energy)
        if Q - to_cs_energy >= 0:
            states.append((i, route[:i+1], Q, departure_time, load, nearest_cs_index, nearest_cs_distance, to_cs_energy,
                           to_cs_time, to_cs_speed, total_travel_time, total_charging_time, total_service_time, insert_cs))
        else:
            insert_cs = True
            # 如果状态列表为空，无法继续,说明应该在上个gb路径最后一个点插入depot
            if not states:
                (new_route, Q, load, departure_time, to_cs_travel_time, to_cs_charging_time,
                 to_cs_service_time) = insert_cs_before_gb(df, dist_matrix, route, Q1, load1, departure_time1)
                total_travel_time += to_cs_travel_time
                total_service_time += to_cs_service_time
                total_charging_time += to_cs_charging_time
                states.append((i, route[:i + 1], Q, departure_time, load, None, None, None, None, None,
                               total_travel_time, total_charging_time, total_service_time, insert_cs))
            else:
                # print('states', states)
                # print('total_travel_time', total_travel_time)
                (prev_i, prev_route, prev_Q, prev_departure_time, prev_load, prev_nearest_cs_index, prev_nearest_cs_distance,
                 prev_to_cs_energy, prev_to_cs_time, prev_to_cs_speed,
                 prev_total_travel_time, prev_total_charging_time, prev_total_service_time, insert_cs) = states[-1]

                total_travel_time = prev_total_travel_time + prev_to_cs_time
                total_service_time = prev_total_service_time

                q_ik = Q_max - (prev_Q - prev_to_cs_energy)
                charging_time = calculate_charging_time(q_ik)
                total_charging_time += charging_time

                route.insert(prev_i + 1, prev_nearest_cs_index)
                departure_time = prev_departure_time + prev_to_cs_time + charging_time
                Q = Q_max
                load = prev_load

                # print('insert_cscscscscs', route[prev_i], route[prev_i+1])
                # print('energy, t_ijk, speed', prev_to_cs_energy, prev_to_cs_time, total_travel_time, total_service_time,
                #       departure_time)

                states.append((i, route[:i+1], Q, departure_time, load, None, None, None, None, None,
                               total_travel_time, total_charging_time, total_service_time, insert_cs))
                i = prev_i + 1

        i += 1
    # print('*****************************************************************************************')
    # print('states', len(states), states)
    # print('route', len(route), route)
    # print(route, Q, load, departure_time, total_travel_time, total_charging_time, total_service_time, insert_cs)
    return states


class ACA_object:
    def __init__(self, nodes_list, distance_matrix,
                 size_ants=10, max_iter=20,
                 alpha=1, beta=2, rho=0.1, epsilon=0.1):
        self.nodes_list = nodes_list
        self.nodes_sum = len(nodes_list)  # 城市数量
        self.ants_sum = size_ants  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度
        self.epsilon = epsilon  # 精英蚂蚁的信息素因子

        self.matrix_distance = distance_matrix[np.ix_(nodes_list, nodes_list)]
        self.matrix_heuristic = 1 / (self.matrix_distance + 1e-10 * np.eye(self.nodes_sum, self.nodes_sum))  # 避免除零错误

        self.matrix_pheromone = np.ones((self.nodes_sum, self.nodes_sum))  # 信息素矩阵，每次迭代都会更新
        self.ants_route = np.zeros((size_ants, self.nodes_sum)).astype(int)  # 某一代每个蚂蚁的爬行路径

        self.ants_cost = None  # 某一代每个蚂蚁的爬行总距离
        self.elite_ant_history, self.elite_ant_cost_history, self.elite_ant_info_history = [], [], []  # 记录各代的最佳情况
        self.phero_elite_ant_cost_history = []
        self.elite_ant_part_cost_history = []
        self.best_route, self.best_cost, self.best_info = None, None, []

    def compute_transition_probabilities(self, current_node, unvisited):
        tau = self.matrix_pheromone[current_node, unvisited]
        eta = self.matrix_heuristic[current_node, unvisited]
        prob = (tau ** self.alpha) * (eta ** self.beta)
        prob /= prob.sum()
        return prob

    def update_pheromone(self, elite_ant, elite_ant_cost, ants_cost):
        # 计算蚂蚁信息素
        delta_tau = np.zeros((self.nodes_sum, self.nodes_sum))
        for j in range(self.ants_sum):  # 每个蚂蚁
            for k in range(self.nodes_sum - 1):  # 每个节点
                n1, n2 = self.ants_route[j, k], self.ants_route[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                delta_tau[n1, n2] += 1 / ants_cost[j]  # 涂抹的信息素
            n1, n2 = self.ants_route[j, self.nodes_sum - 1], self.ants_route[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
            delta_tau[n1, n2] += 1 / ants_cost[j]  # 涂抹信息素

        # 计算精英蚂蚁的信息素
        delta_tau_elite = np.zeros((self.nodes_sum, self.nodes_sum))
        for k in range(self.nodes_sum - 1):
            n1, n2 = elite_ant[k], elite_ant[k + 1]
            delta_tau_elite[n1, n2] += 1 / elite_ant_cost
        n1, n2 = elite_ant[self.nodes_sum - 1], elite_ant[0]
        delta_tau_elite[n1, n2] += 1 / elite_ant_cost

        # 信息素飘散+信息素涂抹
        self.matrix_pheromone = (1 - self.rho) * self.matrix_pheromone + delta_tau + self.epsilon * delta_tau_elite

    def insert_depot(self, df_nodes_idx, para_depot):
        # 在首尾gb中插入depot
        depot_list = df[df['Type'] == 'd'].index.tolist()
        depot_coords = df.loc[depot_list, ['x', 'y']].values.astype('float')
        depots = []
        for idx, ant_route in enumerate(self.ants_route):
            if para_depot == 1:
                customer_coords = df.loc[df_nodes_idx[ant_route[0]], ['x', 'y']].values.astype('float')
            elif para_depot == 2:
                customer_coords = df.loc[df_nodes_idx[ant_route[-1]], ['x', 'y']].values.astype('float')
            else:
                continue

            distances = np.linalg.norm(depot_coords - customer_coords, axis=1)
            nearest_depot_index = depot_list[np.argmin(distances)]
            depots.append(nearest_depot_index)
        return depots

    def run(self, df, dist_matrix, Q, load, departure_time, final_route, next_gbs_center_idx, para_depot, max_iter=None):
        """
        :param final_route:
        :param para_depot: 0: 不插入depot；1:插入初始depot; 2: 插入最终depot
        :param max_iter:
        :return:
        """
        route_phero, travel_time_phero, charging_time_phero, service_time_phero = [], 0, 0, 0
        insert_cs_in_start = False
        self.max_iter = max_iter or self.max_iter
        for _ in range(self.max_iter):  # 对每次迭代

            for j in range(self.ants_sum):  # 对每个蚂蚁
                if para_depot == 1:
                    self.ants_route[j, 0] = np.random.randint(0, self.nodes_sum)  # 随机选择起点
                else:
                    self.ants_route[j, 0] = 0

                # print('self.ants_route[j]', para_depot, self.ants_route[j])
                for k in range(self.nodes_sum - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.ants_route[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    unvisited = list(set(range(self.nodes_sum)) - taboo_set)  # 在这些点中做选择

                    # 选择下个节点
                    prob = self.compute_transition_probabilities(self.ants_route[j, k], unvisited)
                    if np.random.rand() <= 0.5:  # 0.5是rand_0的假设值，可以调整
                        next_point = unvisited[np.argmax(prob)]
                    else:
                        next_point = np.random.choice(unvisited, size=1, p=prob)[0]
                    self.ants_route[j, k + 1] = next_point

            # 计算cost
            ants_cost, ants_part_cost = [], []
            ants_cost_phero = []
            Q_load_departure = []
            route_real_list = []

            depots = self.insert_depot(self.nodes_list, para_depot)
            for idx, ant_route in enumerate(self.ants_route):
                ant_route = [self.nodes_list[i] for i in ant_route]
                # 判断在首尾gb中是否需要插入depot
                if para_depot == 1:
                    ant_route.insert(0, depots[idx])
                    ant_route.append(next_gbs_center_idx)
                elif para_depot == 2:
                    ant_route.append(depots[idx])
                else:
                    ant_route.append(next_gbs_center_idx)


                result_states = routing_time(df, dist_matrix, ant_route, Q, load, departure_time, Q, load, departure_time)

                # if result_states is None:
                #     print('·········································································')
                #     print(final_route)
                #     print(ant_route)
                #     (_, route_real, Q_real, departure_time_real, load_real, _, _, _, _, _, travel_time_real,
                #      charging_time_real, service_time_real, insert_cs_real) = insert_cs_before_gb(df, dist_matrix, ant_route, Q, load, departure_time)
                #     insert_cs_in_start = insert_cs_real
                #     Q_load_departure.append([Q_real, load_real, departure_time_real])
                # else:
                # print('不需要回溯')
                (_, route_phero, Q_phero, departure_time_phero, load_phero, _, _, _, _, _, travel_time_phero,
                 charging_time_phero, service_time_phero, insert_cs_phero) = result_states[-1]
                if para_depot == 2:
                    (_, route_real, Q_real, departure_time_real, load_real, _, _, _, _, _, travel_time_real, charging_time_real,
                     service_time_real, insert_cs_real) = result_states[-1]
                else:
                    (_, route_real, Q_real, departure_time_real, load_real, _, _, _, _, _, travel_time_real,
                     charging_time_real,
                     service_time_real, insert_cs_real) = result_states[-2]
                Q_load_departure.append([Q_real, load_real, departure_time_real])
                # print('result_states[-2]', result_states[-2])
                # print('result_states[-1]', result_states[-1])

                route_real_list.append(route_real)
                cost, part_cost = total_cost(df, route_real, travel_time_real, charging_time_real, service_time_real)
                ants_cost.append(cost)
                ants_part_cost.append(part_cost)
                # print(route_real)
                # print('111111111', route_real_list)

                # 为了更新pheromones，计算组合gb中心路径的cost
                cost_phero, part_cost_phero = total_cost(df, route_phero, travel_time_phero, charging_time_phero, service_time_phero)
                ants_cost_phero.append(cost_phero)
                # print('@@@@@@@@@@@@@@@@@@@@@@@cost', cost, part_cost)
                # print('@@@@@@@@@@@@@@@@@phero_cost', cost_phero, part_cost_phero)
                # print(self.nodes_list)
                # print('ant_route', ant_route)
                # print('cost', cost, cost_phero)

            # 记录ants_phero历史最好情况,更新信息素
            index_elite_ant_phero = np.array(ants_cost_phero).argmin()
            elite_ant_phero, elite_ant_cost_phero = self.ants_route[index_elite_ant_phero].copy(), ants_cost_phero[index_elite_ant_phero].copy()
            self.phero_elite_ant_cost_history.append(ants_cost_phero[index_elite_ant_phero])
            # break

            # print('index_elite_ant_phero', index_elite_ant_phero, elite_ant_cost_phero, elite_ant_phero)
            # 顺便记录历史最好情况
            index_elite_ant = index_elite_ant_phero
            elite_ant, elite_ant_cost = self.ants_route[index_elite_ant].copy(), ants_cost[index_elite_ant].copy()
            elite_ant_part_cost = ants_part_cost[index_elite_ant]
            elite_ant_info = Q_load_departure[index_elite_ant]
            complete_elite_ant = route_real_list[index_elite_ant]
            # print('______index_elite_ant', index_elite_ant, elite_ant_cost, elite_ant_part_cost, complete_elite_ant)

            # 更新信息素
            if route_phero == [] and travel_time_phero == 0 and charging_time_phero == 0 and service_time_phero == 0:
                self.update_pheromone(elite_ant, elite_ant_cost, ants_cost)
            else:
                self.update_pheromone(elite_ant_phero, elite_ant_cost_phero, ants_cost_phero)

            elite_ant_route = complete_elite_ant
            self.elite_ant_history.append(elite_ant_route)
            self.elite_ant_cost_history.append(elite_ant_cost)
            self.elite_ant_part_cost_history.append(elite_ant_part_cost)
            self.elite_ant_info_history.append(elite_ant_info)


        best_generation = np.array(self.phero_elite_ant_cost_history).argmin()
        # print('best_generation', best_generation, self.elite_ant_cost_history)
        self.best_route = self.elite_ant_history[best_generation]
        self.best_cost = (self.elite_ant_cost_history[best_generation], self.elite_ant_part_cost_history[best_generation])
        self.best_info = self.elite_ant_info_history[best_generation]
        # print('self.best_route', self.best_route)
        # print(self.best_cost)
        # print('best_zbest_zbest_z', self.best_info)
        return self.best_route, self.best_cost, self.best_info, insert_cs_in_start

    fit = run



def gb_routing(df, gbs_list, total_dist_matrix, gbs_center_idx, size_ants, max_iter, Q_max=40):
    final_route, final_cost_list, final_part_cost_list = [], [], []
    final_Q = []

    # Initialize Parameters
    Q = Q_max
    departure_time = 0
    nodes_list = [item for sublist in gbs_list for item in sublist]
    load = sum(df.loc[nodes_list, 'demand'])

    # 依次对每个gb内进行规划
    idx = 0
    while idx < len(gbs_list):
        current_gb = gbs_list[idx]
        # 将下一个gb的中心作为最后一个元素
        if idx == 0:
            para_depot = 1  # 首个粒球需要插入depot
            gb = current_gb
            next_gb_center = gbs_center_idx[idx+1]

        elif idx == len(gbs_list) - 1:
            para_depot = 2  # 最后粒球需要插入depot
            gb = final_route[-1:] + current_gb
            next_gb_center = None
        else:
            para_depot = 0
            gb = final_route[-1:] + current_gb
            next_gb_center = gbs_center_idx[idx + 1]


        aca_object = ACA_object(gb, total_dist_matrix, size_ants, max_iter)


        gb_route, gb_cost, next_gb_info, insert_cs_in_start = aca_object.run(df, total_dist_matrix, Q, load, departure_time, final_route, next_gb_center, para_depot)
        gb_total_cost, gb_part_cost = gb_cost
        Q, load, departure_time = next_gb_info


        final_cost_list.append(gb_total_cost)
        final_part_cost_list.append(gb_part_cost)
        final_Q.append(Q)

        # 若在run中插入了depot，则最后在final_route中插入depot
        if para_depot == 1 or insert_cs_in_start == True:
            final_route.extend(gb_route[:])
        elif para_depot == 0:
            final_route.extend(gb_route[1:])
        else:
            final_route.extend(gb_route[1:])


        # print('final_route', final_route)

        idx += 1
        # # 绘制每次gb更新后的路径，一个gb绘制一次
        # plot_points(df, final_route)

    final_cost = sum(final_cost_list)
    final_part_cost = [sum(column) for column in zip(*final_part_cost_list)]
    return final_route, final_cost, final_part_cost, final_Q[-1]


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
    # # c101
    # df = pd.read_csv(r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\c101_21_service10.txt",
    #     sep=r'\s+')
    # clusters = {
    #     0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 111, 112, 113,
    #         114, 115, 116, 117, 118, 119, 120, 121},
    #     1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #         68, 69, 70, 71, 72, 73},
    #     2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100,
    #         101, 102, 103, 104, 105, 107, 108}}

    # # r101
    # df = pd.read_csv(
    #     r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\r101_21.txt',
    #     sep=r'\s+')
    # clusters = {0: {24, 25, 33, 42, 43, 44, 45, 46, 47, 48, 49, 50, 60, 61, 71, 73, 74, 75, 76, 77, 79, 88, 89, 90, 93, 94, 95, 96, 97, 98, 100, 101, 110}, 1: {23, 26, 27, 29, 34, 35, 36, 37, 38, 39, 58, 59, 62, 63, 64, 65, 66, 67, 78, 80, 81, 82, 104, 105, 106, 107, 108, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121}, 2: {22, 28, 30, 31, 32, 40, 41, 51, 52, 53, 54, 55, 56, 57, 68, 69, 70, 72, 83, 84, 85, 86, 87, 91, 92, 99, 102, 103, 109, 111}}

    # RC101
    df = pd.read_csv(
        r"C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\table 1\rc101_21.txt",
        sep=r'\s+')
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

    W = 650
    Q_max = 40

    # 记录运行时间
    start_time = time.time()

    k = 1
    gbs_list, gbs_center_location = generate_gbs(df, clusters, k)

    # 将gb_centers增加到df中
    df, gb_centers_df_idx = add_gb_centers_in_dataframe(df, gbs_center_location)
    total_dist_matrix = total_distance_matrix(df)

    # 在GBs中进行EVRP
    routes = []
    cost_cluster, part_cost_cluster = [], []
    remaining_Q = []
    for i, cluster in list(gbs_list.items())[2:]:
        centers1 = np.array(gbs_center_location[i])
        gbs_distance_matrix = cdist(centers1, centers1, metric='euclidean')
        aca_gb = ACA_GB(func=cal_total_distance, n_dim=len(centers1),
                      size_pop=10, max_iter=30,
                      distance_matrix=gbs_distance_matrix)

        gbs_order, _ = aca_gb.run()
        gbs_list1 = [gbs_list[i][j] for j in gbs_order]
        gbs_center_idx = [gb_centers_df_idx[i][j] for j in gbs_order]

        size_ants, Iter = 10, 100
        final_route, final_cost, final_part_cost, final_Q = gb_routing(df, gbs_list1, total_dist_matrix, gbs_center_idx, size_ants, Iter)
        routes.append(final_route)
        cost_cluster.append(final_cost)
        part_cost_cluster.append(final_part_cost)
        remaining_Q.append(final_Q)

        plot_points(df, final_route)
        # break

    end_time = time.time()
    print(f"Running time: {end_time - start_time:.4f}s")

    routes_dist = get_routes_distances(routes, total_dist_matrix)
    part_cost_cluster_sums = [sum(x) for x in zip(*part_cost_cluster)]
    print('final_route =', routes)
    print('total_cost =', sum(cost_cluster))
    print('part_cost =', part_cost_cluster_sums)
    print('routes_dist =', sum(routes_dist), routes_dist)
    print('remaining_Q =', sum(remaining_Q), remaining_Q)
    print('each_cluster_cost =', cost_cluster, part_cost_cluster)
    plot_final_routes(df, routes)


    # 合并所有子列表
    all_elements = [item for sublist in routes for item in sublist]
    # 找到重复元素
    duplicates = set([item for item in all_elements if all_elements.count(item) > 1])
    # 计算总元素数
    total_elements = len(all_elements)
    # 输出结果
    print("总元素数:", total_elements)
    print("重复元素:", duplicates)
    print("重复元素数:", len(duplicates))

