"""
A*寻路算法实现
包含单向A*和双向A*算法的实现，用于解决实验二中的寻路问题。
"""

import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import time

# 设置中文字体，避免中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


class Node:
    """A*算法的节点类"""

    def __init__(self, position, parent=None):
        """
        初始化节点

        参数:
            position: 节点在网格中的位置，格式为(行, 列)
            parent: 父节点
        """
        self.position = position  # 格式(行, 列)
        self.parent = parent  # 父节点，用于回溯路径

        self.g = 0  # 从起点到该节点的实际代价
        self.h = 0  # 从该节点到终点的估计代价（启发式函数值）
        self.f = 0  # f = g + h，总代价

    def __eq__(self, other):
        """判断两个节点是否相同（位置相同）"""
        return self.position == other.position

    def __lt__(self, other):
        """比较两个节点的优先级（用于优先队列）"""
        return self.f < other.f

    def __repr__(self):
        """返回节点的字符串表示"""
        return f"Node(pos={self.position}, g={self.g:.2f}, h={self.h:.2f}, f={self.f:.2f})"


class AStar:
    """A*寻路算法实现"""

    def __init__(self, grid):
        """
        初始化A*算法

        参数:
            grid: 地图网格，二维数组表示
                0 - 普通地形（地形代价为0）
                1 - 障碍物（不可通行）
                2 - 沙漠（地形代价为4）
                3 - 溪流（地形代价为2）
                4 - 起点S
                5 - 终点T
        """
        self.grid = grid
        self.rows, self.cols = grid.shape

        # 找到起点和终点
        self.start_pos = None
        self.end_pos = None

        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i, j] == 4:  # 起点S
                    self.start_pos = (i, j)
                elif grid[i, j] == 5:  # 终点T
                    self.end_pos = (i, j)

        if self.start_pos is None or self.end_pos is None:
            raise ValueError("地图中必须包含起点(S)和终点(T)")

    def get_terrain_cost(self, position):
        """
        获取指定位置的地形代价

        参数:
            position: 位置坐标(行, 列)

        返回:
            地形代价值
        """
        x, y = position
        terrain_type = self.grid[x, y]

        if terrain_type == 0:  # 普通地形
            return 0
        elif terrain_type == 2:  # 沙漠
            return 4
        elif terrain_type == 3:  # 溪流
            return 2
        elif terrain_type == 4 or terrain_type == 5:  # 起点或终点
            return 0
        else:
            return 0  # 默认地形代价

    def get_neighbors(self, node):
        """
        获取节点的相邻节点

        参数:
            node: 当前节点

        返回:
            相邻节点位置的列表
        """
        x, y = node.position
        neighbors = []

        # 8个方向的移动: 上、右上、右、右下、下、左下、左、左上
        directions = [
            (-1, 0),  # 上
            (-1, 1),  # 右上
            (0, 1),  # 右
            (1, 1),  # 右下
            (1, 0),  # 下
            (1, -1),  # 左下
            (0, -1),  # 左
            (-1, -1)  # 左上
        ]

        for direction in directions:
            # 计算新位置
            new_x = x + direction[0]
            new_y = y + direction[1]

            # 检查新位置是否在地图范围内
            if 0 <= new_x < self.rows and 0 <= new_y < self.cols:
                # 检查新位置是否是障碍物
                if self.grid[new_x, new_y] != 1:
                    neighbors.append((new_x, new_y))

        return neighbors

    def calculate_h(self, position, target):
        """
        计算启发式函数值（欧几里得距离）

        参数:
            position: 当前位置
            target: 目标位置

        返回:
            欧几里得距离
        """
        return math.sqrt((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2)

    def calculate_move_cost(self, current_pos, next_pos):
        """
        计算移动代价

        参数:
            current_pos: 当前位置
            next_pos: 下一位置

        返回:
            移动代价（直线移动为1，对角线移动为√2）
        """
        dx = abs(current_pos[0] - next_pos[0])
        dy = abs(current_pos[1] - next_pos[1])

        if dx == 1 and dy == 1:  # 对角线移动
            return math.sqrt(2)
        else:  # 直线移动
            return 1.0

    def single_direction_astar(self):
        """
        单向A*算法

        返回:
            path: 找到的路径（从起点到终点的位置列表）
            visited: 搜索过程中访问过的节点列表
            cost: 路径总代价
        """
        # 创建起点和终点节点
        start_node = Node(self.start_pos)
        end_node = Node(self.end_pos)

        # 初始化开放列表和关闭集合
        open_list = []  # 优先队列，保存待探索的节点
        closed_set = set()  # 已探索过的节点集合

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)

        # 记录访问过的节点
        visited = []

        # 开始搜索
        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)
            visited.append(current_node.position)

            # 如果当前节点是终点，构建路径并返回
            if current_node.position == end_node.position:
                path = []
                cost = current_node.g  # 记录总代价

                # 回溯构建路径
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent

                return path[::-1], visited, cost  # 反转路径，使其从起点到终点

            # 将当前节点加入关闭集合
            closed_set.add(current_node.position)

            # 探索相邻节点
            for neighbor_pos in self.get_neighbors(current_node):
                # 如果相邻节点已经在关闭集合中，跳过
                if neighbor_pos in closed_set:
                    continue

                # 创建相邻节点
                neighbor_node = Node(neighbor_pos, current_node)

                # 计算从起点到相邻节点的代价
                move_cost = self.calculate_move_cost(current_node.position, neighbor_pos)
                terrain_cost = self.get_terrain_cost(neighbor_pos)
                neighbor_node.g = current_node.g + move_cost + terrain_cost

                # 计算启发式函数值
                neighbor_node.h = self.calculate_h(neighbor_pos, self.end_pos)

                # 计算f值
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # 检查相邻节点是否已经在开放列表中
                # 如果在开放列表中且新路径更好，则更新
                should_add = True
                for i, open_node in enumerate(open_list):
                    if open_node.position == neighbor_node.position:
                        if open_node.g <= neighbor_node.g:
                            should_add = False  # 旧路径更好，不需要更新
                        else:
                            # 新路径更好，替换旧节点
                            open_list[i] = neighbor_node
                            heapq.heapify(open_list)  # 重新排序
                            should_add = False
                        break

                # 如果是新节点，或者是更好的路径，加入开放列表
                if should_add:
                    heapq.heappush(open_list, neighbor_node)

        # 如果开放列表为空仍未找到路径，则返回失败
        return None, visited, float('inf')

    def bidirectional_astar(self):
        """
        双向A*算法
        从起点和终点同时开始搜索，当两个搜索相遇时找到路径

        返回:
            path: 找到的路径（从起点到终点的位置列表）
            forward_visited: 前向搜索访问过的节点列表
            backward_visited: 后向搜索访问过的节点列表
            cost: 路径总代价
        """
        # 创建起点和终点节点
        start_node = Node(self.start_pos)
        end_node = Node(self.end_pos)

        # 初始化前向搜索的开放列表、关闭集合和节点字典
        forward_open = []  # 前向搜索的开放列表
        forward_closed = set()  # 前向搜索的关闭集合
        forward_nodes = {self.start_pos: start_node}  # 前向搜索的节点字典

        # 初始化后向搜索的开放列表、关闭集合和节点字典
        backward_open = []  # 后向搜索的开放列表
        backward_closed = set()  # 后向搜索的关闭集合
        backward_nodes = {self.end_pos: end_node}  # 后向搜索的节点字典

        # 将起点和终点加入各自的开放列表
        heapq.heappush(forward_open, start_node)
        heapq.heappush(backward_open, end_node)

        # 记录访问过的节点
        forward_visited = []
        backward_visited = []

        # 记录最佳路径的中间节点和代价
        best_cost = float('inf')
        meeting_point = None

        # 当前向搜索和后向搜索都还有节点可以扩展时
        while forward_open and backward_open:
            # 处理前向搜索
            if forward_open:
                # 获取前向搜索中f值最小的节点
                current_forward = heapq.heappop(forward_open)
                forward_visited.append(current_forward.position)

                # 检查当前前向节点是否已经在后向搜索中被访问过
                if current_forward.position in backward_nodes:
                    # 找到了路径，计算总代价
                    backward_node = backward_nodes[current_forward.position]
                    total_cost = current_forward.g + backward_node.g

                    # 如果这条路径更好，更新最佳路径
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = current_forward.position

                # 将当前节点加入前向搜索的关闭集合
                forward_closed.add(current_forward.position)

                # 探索前向搜索的相邻节点
                for neighbor_pos in self.get_neighbors(current_forward):
                    # 如果相邻节点已经在前向搜索的关闭集合中，跳过
                    if neighbor_pos in forward_closed:
                        continue

                    # 计算新路径的代价
                    move_cost = self.calculate_move_cost(current_forward.position, neighbor_pos)
                    terrain_cost = self.get_terrain_cost(neighbor_pos)
                    g_score = current_forward.g + move_cost + terrain_cost

                    # 检查是否已经有更好的路径
                    if neighbor_pos in forward_nodes and g_score >= forward_nodes[neighbor_pos].g:
                        continue

                    # 创建或更新相邻节点
                    neighbor_node = Node(neighbor_pos, current_forward)
                    neighbor_node.g = g_score
                    neighbor_node.h = self.calculate_h(neighbor_pos, self.end_pos)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    # 更新节点字典
                    forward_nodes[neighbor_pos] = neighbor_node

                    # 将节点加入前向搜索的开放列表
                    heapq.heappush(forward_open, neighbor_node)

            # 处理后向搜索
            if backward_open:
                # 获取后向搜索中f值最小的节点
                current_backward = heapq.heappop(backward_open)
                backward_visited.append(current_backward.position)

                # 检查当前后向节点是否已经在前向搜索中被访问过
                if current_backward.position in forward_nodes:
                    # 找到了路径，计算总代价
                    forward_node = forward_nodes[current_backward.position]
                    total_cost = forward_node.g + current_backward.g

                    # 如果这条路径更好，更新最佳路径
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = current_backward.position

                # 将当前节点加入后向搜索的关闭集合
                backward_closed.add(current_backward.position)

                # 探索后向搜索的相邻节点
                for neighbor_pos in self.get_neighbors(current_backward):
                    # 如果相邻节点已经在后向搜索的关闭集合中，跳过
                    if neighbor_pos in backward_closed:
                        continue

                    # 计算新路径的代价
                    move_cost = self.calculate_move_cost(current_backward.position, neighbor_pos)
                    terrain_cost = self.get_terrain_cost(neighbor_pos)
                    g_score = current_backward.g + move_cost + terrain_cost

                    # 检查是否已经有更好的路径
                    if neighbor_pos in backward_nodes and g_score >= backward_nodes[neighbor_pos].g:
                        continue

                    # 创建或更新相邻节点
                    neighbor_node = Node(neighbor_pos, current_backward)
                    neighbor_node.g = g_score
                    neighbor_node.h = self.calculate_h(neighbor_pos, self.start_pos)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    # 更新节点字典
                    backward_nodes[neighbor_pos] = neighbor_node

                    # 将节点加入后向搜索的开放列表
                    heapq.heappush(backward_open, neighbor_node)

            # 检查终止条件：如果已经找到路径，并且两个方向搜索的最小f值之和大于最佳路径的代价，就可以结束搜索
            if meeting_point and forward_open and backward_open:
                min_forward_f = forward_open[0].f
                min_backward_f = backward_open[0].f
                if min_forward_f + min_backward_f >= best_cost:
                    break

        # 如果找到了路径，构建完整路径
        if meeting_point:
            # 构建从起点到相遇点的路径
            forward_path = []
            current = forward_nodes[meeting_point]
            while current:
                forward_path.append(current.position)
                current = current.parent
            forward_path = forward_path[::-1]  # 反转，使其从起点到相遇点

            # 构建从终点到相遇点的路径
            backward_path = []
            current = backward_nodes[meeting_point]
            if current.parent:  # 跳过相遇点本身，因为它已经在forward_path中
                current = current.parent
                while current:
                    backward_path.append(current.position)
                    current = current.parent

            # 合并路径，得到从起点到终点的完整路径
            complete_path = forward_path + backward_path

            return complete_path, forward_visited, backward_visited, best_cost

        # 如果没有找到路径，返回失败
        return None, forward_visited, backward_visited, float('inf')


def create_example_map_1():
    """
    创建图1所示的示例地图

    返回:
        grid: 二维数组表示的地图
    """
    # 初始化一个20x20的地图，所有格子都是普通地形
    grid = np.zeros((20, 20), dtype=int)

    # 设置障碍物（灰色格子）
    obstacles = [(8, 7), (9, 7), (10, 8), (11, 8), (12, 8), (12, 9), (13, 9), (14, 9)]
    for x, y in obstacles:
        grid[x, y] = 1

    # 设置起点和终点
    grid[11, 5] = 4  # 起点 S
    grid[12, 14] = 5  # 终点 T

    return grid


def create_example_map_2():
    """
    创建图2所示的示例地图

    返回:
        grid: 二维数组表示的地图
    """
    # 初始化一个20x40的地图，所有格子都是普通地形
    grid = np.zeros((20, 40), dtype=int)

    # 设置障碍物（灰色格子）
    obstacles = [
        (0, 3), (0, 7), (0, 12),
        (1, 7), (1, 12),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 7), (2, 8), (2, 9), (2, 10), (2, 12),
        (3, 8), (3, 12),
        (4, 12),
        (5, 7), (5, 8), (5, 12),
        (6, 2), (6, 3), (6, 4), (6, 5), (6,6), (6, 7), (6, 12),
        (7, 2), (7, 5), (7, 7), (7, 12), (7, 36),
        (8, 5),
        (9, 5), (9, 7), (9, 36),
        (10, 2), (10, 5), (10, 7), (10, 8), (10, 19), (10, 20), (10, 21), (10, 28),
        (11, 2), (11, 3), (11, 4), (11, 5), (11, 8), (11, 19), (11, 20), (11, 21), (11, 31),
        (12, 3), (12, 8), (12, 12), (12, 19), (12, 20), (12, 21),
        (13, 3), (13, 8), (13, 9), (13, 11), (13, 12), (13, 31),
        (14, 3), (14, 8), (14, 12),
        (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 12), (15, 24), (15, 25),
        (16, 3), (16, 12), (16, 24), (16, 25),
        (17, 7), (17, 12),
        (18, 3), (18, 7), (18, 12),
        (19, 3), (19, 7), (19, 12),
    ]
    for x, y in obstacles:
        grid[x, y] = 1

    # 设置沙漠（黄色格子，代价为4）
    desert = []
    for i in range(24, 40):
        desert.append((0, i))
    for i in range(25, 40):
        desert.append((1, i))
    for i in range(26, 40):
        desert.append((2, i))
    for i in range(26, 37):
        desert.append((3, i))
    for i in range(26, 36):
        desert.append((4, i))
    for i in range(27, 33):
        desert.append((5, i))
    for i in range(27, 33):
        desert.append((6, i))
    for i in range(29, 33):
        desert.append((7, i))

    for x, y in desert:
        grid[x, y] = 2

    # 设置溪流（蓝色格子，代价为2）
    river = [
        (1, 34),
        (2, 33),
        (3, 32),
        (4, 33),
        (5, 33), (5, 34),
        (6, 33), (6, 34),
        (7, 33), (7, 34), (7, 35),
        (8, 32), (8, 33), (8, 34), (8, 35),
        (9, 32), (9, 33), (9, 34),
        (10, 32), (10, 33), (10, 35), (10, 36),
        (11, 32), (11, 34), (11, 35),
        (12, 33), (12, 34),
        (13, 32), (13, 33), (13, 34),
        (14, 32), (14, 33), (14, 34),
        (15, 31), (15, 32), (15, 33),
        (16, 31), (16, 32), (16, 33),
        (17, 30), (17, 31), (17, 32),
        (18, 29), (18, 30), (18, 31),
        (19, 28), (19, 29), (19, 30),
    ]

    for x, y in river:
        grid[x, y] = 3

    # 设置起点和终点
    grid[10, 4] = 4  # 起点 S
    grid[0, 35] = 5  # 终点 T

    return grid


def visualize_map(grid, path=None, forward_visited=None, backward_visited=None, title="A*寻路算法可视化"):
    """
    可视化地图、路径和搜索过程

    参数:
        grid: 地图网格
        path: 找到的路径
        forward_visited: 前向搜索访问过的节点
        backward_visited: 后向搜索访问过的节点
        title: 图像标题

    返回:
        fig: matplotlib图像对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建颜色映射
    colors = ['white', 'gray', 'yellow', 'skyblue', 'green', 'red']
    cmap = ListedColormap(colors)

    # 绘制地图
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=5)

    # 绘制网格线
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)

    # 找到并标记起点和终点
    start_pos = None
    end_pos = None
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 4:  # 起点S
                start_pos = (i, j)
                ax.text(j, i, 'S', ha='center', va='center', fontsize=12, color='black', fontweight='bold')
            elif grid[i, j] == 5:  # 终点T
                end_pos = (i, j)
                ax.text(j, i, 'T', ha='center', va='center', fontsize=12, color='black', fontweight='bold')

    # 绘制前向搜索访问过的节点
    if forward_visited:
        for pos in forward_visited:
            i, j = pos
            if grid[i, j] != 4 and grid[i, j] != 5:  # 不在起点和终点上绘制
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='orangered', linewidth=1, alpha=0.5))

    # 绘制后向搜索访问过的节点
    if backward_visited:
        for pos in backward_visited:
            i, j = pos
            if grid[i, j] != 4 and grid[i, j] != 5:  # 不在起点和终点上绘制
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=1, alpha=0.5))

    # 绘制路径
    if path:
        # 转换路径格式为[(y, x)]用于绘图
        path_points = [(j, i) for i, j in path]

        # 绘制路径线段
        for i in range(len(path_points) - 1):
            ax.plot([path_points[i][0], path_points[i + 1][0]],
                    [path_points[i][1], path_points[i + 1][1]],
                    color='black', linewidth=2)

        # 在路径上绘制点
        for i, j in path:
            if (i, j) != start_pos and (i, j) != end_pos:  # 不在起点和终点上绘制
                ax.plot(j, i, 'o', color='black', markersize=6)

    # 设置坐标轴
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))

    # 确保正确的宽高比
    ax.set_aspect('equal')

    # 添加图例
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='white', label='普通地形 (代价: 0)'),
        Rectangle((0, 0), 1, 1, color='gray', label='障碍物 (不可通行)'),
        Rectangle((0, 0), 1, 1, color='yellow', label='沙漠 (代价: 4)'),
        Rectangle((0, 0), 1, 1, color='skyblue', label='溪流 (代价: 2)'),
        Rectangle((0, 0), 1, 1, color='green', label='起点 (S)'),
        Rectangle((0, 0), 1, 1, color='red', label='终点 (T)'),
        Line2D([0], [0], marker='o', color='black', label='路径', markersize=6, linewidth=2),
    ]

    if forward_visited and not backward_visited:
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor='orangered', label='访问节点')
        )
    elif forward_visited and backward_visited:
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor='orangered', label='前向搜索')
        )
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', label='后向搜索')
        )

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=10)

    plt.title(title, fontsize=14)
    plt.tight_layout()

    return fig


def run_example_1():
    """
    运行图1示例
    """
    print("\n===== 运行图1示例 =====")

    # 创建地图
    grid = create_example_map_1()

    # 创建A*算法对象
    astar = AStar(grid)

    # 运行单向A*算法
    print("单向A*算法开始...")
    start_time = time.time()
    single_path, single_visited, single_cost = astar.single_direction_astar()
    end_time = time.time()
    print(f"单向A*算法完成，用时: {end_time - start_time:.6f}秒")

    if single_path:
        print(f"找到路径，长度: {len(single_path)}, 总代价: {single_cost:.2f}")
    else:
        print("未找到路径")

    # 可视化单向A*结果
    fig1 = visualize_map(grid, single_path, single_visited, title="图1 - 单向A*算法")
    fig1.savefig("map1_single_astar.png", dpi=300, bbox_inches='tight')

    # 运行双向A*算法
    print("\n双向A*算法开始...")
    start_time = time.time()
    bi_path, forward_visited, backward_visited, bi_cost = astar.bidirectional_astar()
    end_time = time.time()
    print(f"双向A*算法完成，用时: {end_time - start_time:.6f}秒")

    if bi_path:
        print(f"找到路径，长度: {len(bi_path)}, 总代价: {bi_cost:.2f}")
    else:
        print("未找到路径")

    # 可视化双向A*结果
    fig2 = visualize_map(grid, bi_path, forward_visited, backward_visited, title="图1 - 双向A*算法")
    fig2.savefig("map1_bidirectional_astar.png", dpi=300, bbox_inches='tight')

    plt.show()


def run_example_2():
    """
    运行图2示例
    """
    print("\n===== 运行图2示例 =====")

    # 创建地图
    grid = create_example_map_2()

    # 创建A*算法对象
    astar = AStar(grid)

    # 运行单向A*算法
    print("单向A*算法开始...")
    start_time = time.time()
    single_path, single_visited, single_cost = astar.single_direction_astar()
    end_time = time.time()
    print(f"单向A*算法完成，用时: {end_time - start_time:.6f}秒")

    if single_path:
        print(f"找到路径，长度: {len(single_path)}, 总代价: {single_cost:.2f}")
    else:
        print("未找到路径")

    # 可视化单向A*结果
    fig1 = visualize_map(grid, single_path, single_visited, title="图2 - 单向A*算法")
    fig1.savefig("map2_single_astar.png", dpi=300, bbox_inches='tight')

    # 运行双向A*算法
    print("\n双向A*算法开始...")
    start_time = time.time()
    bi_path, forward_visited, backward_visited, bi_cost = astar.bidirectional_astar()
    end_time = time.time()
    print(f"双向A*算法完成，用时: {end_time - start_time:.6f}秒")

    if bi_path:
        print(f"找到路径，长度: {len(bi_path)}, 总代价: {bi_cost:.2f}")
    else:
        print("未找到路径")

    # 可视化双向A*结果
    fig2 = visualize_map(grid, bi_path, forward_visited, backward_visited, title="图2 - 双向A*算法")
    fig2.savefig("map2_bidirectional_astar.png", dpi=300, bbox_inches='tight')

    plt.show()


def main():
    """
    主函数，用户选择运行示例
    """
    print("===== A*寻路算法实验 =====")
    print("1. 运行图1示例")
    print("2. 运行图2示例")
    print("3. 运行两个示例")

    choice = input("请选择要运行的示例 (1/2/3): ")

    if choice == '1':
        run_example_1()
    elif choice == '2':
        run_example_2()
    elif choice == '3':
        run_example_1()
        print("\n" + "=" * 30 + "\n")
        run_example_2()
    else:
        print("无效的选择，请输入1、2或3")


if __name__ == "__main__":
    main()