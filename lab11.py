import math
import random
import time
import matplotlib.pyplot as plt
from functools import cmp_to_key

# --- Matplotlib 中文显示设置 ---
# 尝试设置支持中文的字体，例如 'SimHei' 或 'Microsoft YaHei'
# 请确保你的系统中安装了这些字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"无法设置中文字体 'SimHei'，绘图可能出现中文乱码: {e}")
    print("请尝试安装 'SimHei' 或 'Microsoft YaHei' 字体，或修改代码使用您系统支持的中文字体。")


# --- 数据结构与辅助函数 ---

class Point:
    """表示平面上的一个点"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        # 用于打印点对象时的表示
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        # 判断两个点是否相等
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        # 允许将点放入集合或字典中
        return hash((self.x, self.y))

def orientation(p, q, r):
    """
    计算三个点 p, q, r 的方向。
    返回值:
        0 --> p, q, r 共线
        1 --> 顺时针
        2 --> 逆时针
    """
    # 添加类型检查以增加健壮性
    if not all(isinstance(pt, Point) for pt in [p, q, r]):
        raise TypeError("Input must be Point objects")
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    # 使用小的容差处理浮点数比较
    epsilon = 1e-9
    if abs(val) < epsilon: return 0  # 共线
    return 1 if val > 0 else 2  # 顺时针或逆时针

def dist_sq(p1, p2):
    """计算两点之间距离的平方"""
    if not all(isinstance(pt, Point) for pt in [p1, p2]):
        raise TypeError("Input must be Point objects")
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

def generate_random_points(n, min_coord=0, max_coord=100):
    """在指定正方形区域内生成 n 个随机点"""
    points = set() # 使用集合避免重复点
    while len(points) < n:
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        points.add(Point(x, y))
    return list(points)

# --- 算法实现 ---

# 4.1 基于枚举方法的凸包求解算法 (O(n^3) Brute Force)
def convex_hull_brute_force(points):
    """
    使用枚举法（暴力法）计算凸包。
    检查每对点 (p, q)，看是否所有其他点都在线段 pq 的同一侧。
    时间复杂度: O(n^3)
    """
    n = len(points)
    if n < 3:
        return sorted(points, key=lambda p: (p.x, p.y)) # 点数少于3，按x,y排序返回

    hull_points = set() # 使用集合存储凸包顶点，自动处理重复

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            p = points[i]
            q = points[j]
            is_edge = True
            all_on_one_side = True
            first_side = None # 记录第一个非共线点的方向

            for k in range(n):
                if k == i or k == j:
                    continue
                r = points[k]
                o = orientation(p, q, r)

                if o == 0: # 共线点
                    # 检查共线点 r 是否在 p 和 q 之间 (包含端点)
                    # 如果 r 在线段 pq 之外的延长线上，则 pq 不是边
                    px, py = p.x, p.y
                    qx, qy = q.x, q.y
                    rx, ry = r.x, r.y
                    dot_product_pq_pr = (qx - px) * (rx - px) + (qy - py) * (ry - py)
                    dot_product_qp_qr = (px - qx) * (rx - qx) + (py - qy) * (ry - qy)

                    # 如果点积 > 0，表示向量方向大致相同
                    # 如果点积 <= 0，表示 r 在线段 pq 内部或端点或反向延长线上
                    # 如果 r 在线段 pq 的延长线上且在 pq 之外
                    if dot_product_pq_pr > dist_sq(p,q) or dot_product_qp_qr > dist_sq(p,q):
                         is_edge = False
                         break
                    # 如果 r 在 p 和 q 之间，暂时不影响 is_edge 判断，最后再加入 hull
                    continue # 继续检查下一个点k

                # 处理非共线点
                if first_side is None:
                    first_side = o # 记录第一个点的方向 (1 或 2)
                elif o != first_side:
                    all_on_one_side = False
                    break # 发现有点在另一侧，pq 不是边

            if is_edge and all_on_one_side:
                hull_points.add(p)
                hull_points.add(q)

    # 处理所有点共线的情况
    if not hull_points and n >= 1:
        if n <= 2: return sorted(points, key=lambda p: (p.x, p.y))
        # 检查是否所有点都共线
        p0 = points[0]
        p1 = points[1]
        all_collinear = True
        for k in range(2, n):
            if orientation(p0, p1, points[k]) != 0:
                all_collinear = False
                break
        if all_collinear:
            # 如果所有点共线，返回x最小和x最大的点
            min_p = min(points, key=lambda p: (p.x, p.y))
            max_p = max(points, key=lambda p: (p.x, p.y))
            return [min_p, max_p] if min_p != max_p else [min_p]
        # 如果并非所有点共线，但之前的循环没找到边，说明实现可能有误或数据特殊
        # 这种情况理论上不应发生，除非点集非常特殊（如只有一个点）
        # 但为了健壮性，可以返回一个基于 Graham Scan 的结果
        if n>0: return convex_hull_graham_scan(points.copy())
        else: return []


    # 对凸包点进行排序 (通常按逆时针)
    hull_list = list(hull_points)
    if not hull_list:
        return [] # 如果没有找到凸包点（例如输入为空）

    # 找到 y 最小的点，如果 y 相同则 x 最小的点作为起点
    start_idx = 0
    for i in range(1, len(hull_list)):
        if hull_list[i].y < hull_list[start_idx].y or \
           (hull_list[i].y == hull_list[start_idx].y and hull_list[i].x < hull_list[start_idx].x):
            start_idx = i
    p0 = hull_list.pop(start_idx)

    # 根据相对于 p0 的极角排序
    hull_list.sort(key=cmp_to_key(lambda p1, p2: compare_points(p0, p1, p2)))

    return [p0] + hull_list


# 4.2 基于 Graham-Scan 的凸包求解算法 (O(n log n))
def convex_hull_graham_scan(points):
    """
    使用 Graham Scan 算法计算凸包。
    时间复杂度: O(n log n)
    """
    points_copy = points[:] # 创建副本，避免修改原始列表
    n = len(points_copy)
    if n < 3:
        return sorted(points_copy, key=lambda p: (p.x, p.y)) # 点数少于3，按x,y排序返回

    # 1. 找到 y 坐标最小的点 p0 (如果 y 相同，则取 x 最小的点)
    min_idx = 0
    for i in range(1, n):
        if points_copy[i].y < points_copy[min_idx].y or \
           (points_copy[i].y == points_copy[min_idx].y and points_copy[i].x < points_copy[min_idx].x):
            min_idx = i
    points_copy[0], points_copy[min_idx] = points_copy[min_idx], points_copy[0]
    p0 = points_copy[0]

    # 2. 根据相对于 p0 的极角对剩余点进行排序
    #    如果极角相同，则距离近的点排在前面 (compare_points 处理)
    sorted_points = sorted(points_copy[1:], key=cmp_to_key(lambda p1, p2: compare_points(p0, p1, p2)))

    # 3. 处理共线点：如果多个点与 p0 共线且角度相同，只保留最远的点
    #    这一步对于 Graham Scan 的栈操作不是必需的，但可以简化栈操作逻辑
    #    栈操作本身会处理掉中间的共线点
    unique_angle_points = []
    if sorted_points: # 确保列表非空
        last_unique = sorted_points[0]
        unique_angle_points.append(last_unique)
        for i in range(1, len(sorted_points)):
            current = sorted_points[i]
            # 如果当前点与上一个加入 unique_angle_points 的点相对于 p0 共线
            o = orientation(p0, last_unique, current)
            if o == 0:
                # 保留距离 p0 更远的点
                if dist_sq(p0, current) > dist_sq(p0, last_unique):
                    unique_angle_points.pop() # 移除较近的点
                    unique_angle_points.append(current) # 添加较远的点
                    last_unique = current # 更新最后一个唯一角度点
            else:
                unique_angle_points.append(current)
                last_unique = current

    # 4. 构建凸包
    if len(unique_angle_points) < 2: # 至少需要 p0 和另外两个非共线点才能形成非退化凸包
         # 如果所有点共线 (unique_angle_points 只有一个元素)
         if len(unique_angle_points) == 1 and all(orientation(p0, unique_angle_points[0], p) == 0 for p in points_copy[1:]):
              # 返回x或y范围最大的两个点
              min_p = min(points_copy, key=lambda p: (p.x, p.y))
              max_p = max(points_copy, key=lambda p: (p.x, p.y))
              return [min_p, max_p] if min_p != max_p else [min_p]
         elif len(unique_angle_points) == 0: # 只有p0一个点
             return [p0]
         else: # 至少有p0和一个其他点，但不足以形成多边形
             return [p0] + unique_angle_points


    hull = [p0, unique_angle_points[0]] # 初始栈包含 p0 和第一个排序后的点

    # 如果只有两个唯一角度点，检查它们是否与p0共线
    if len(unique_angle_points) >= 2:
         hull.append(unique_angle_points[1])
         for i in range(2, len(unique_angle_points)):
             top = hull.pop()
             # 当新点 unique_angle_points[i] 加入时，如果导致向右转（顺时针或共线），则弹出栈顶元素
             # 保持逆时针方向
             while len(hull) >= 1 and orientation(hull[-1], top, unique_angle_points[i]) != 2: # 2 表示严格逆时针
                 top = hull.pop()
                 if len(hull) == 0: break # 防止空栈访问
             hull.append(top)
             hull.append(unique_angle_points[i])
    elif len(unique_angle_points) == 1: # 只有p0和另一个点
        pass # hull 已经是 [p0, unique_angle_points[0]]

    # 最后的检查：如果最后一个点和第一个点与栈顶第二个点共线，移除最后一个点
    # （这种情况较少见，但在某些共线场景下可能需要）
    # while len(hull) >= 3 and orientation(hull[-2], hull[-1], hull[0]) != 2:
    #      hull.pop()

    return hull


def compare_points(p0, p1, p2):
    """
    用于 Graham Scan 排序的比较函数。
    根据相对于 p0 的极角排序。共线时，距离近的优先。
    """
    o = orientation(p0, p1, p2)
    if o == 0: # 共线
        # 距离 p0 近的点排在前面
        d1 = dist_sq(p0, p1)
        d2 = dist_sq(p0, p2)
        # 根据距离排序，距离小的在前
        if d1 < d2: return -1
        if d1 > d2: return 1
        return 0 # 距离相等（理论上随机点不会发生）
    # p1 在 p2 的逆时针方向时返回 -1
    return -1 if o == 2 else 1


# 4.3 基于分治思想的凸包求解算法 (O(n log n))
def convex_hull_divide_conquer(points):
    """
    使用分治法计算凸包 (通常指 MergeHull 或 Andrew's Monotone Chain)。
    这里实现 MergeHull 的思路。
    时间复杂度: O(n log n)
    """
    points_copy = sorted(points, key=lambda p: p.x) # 按 x 坐标排序
    n = len(points_copy)

    if n <= 5: # 对于小规模子问题，使用 Graham Scan (通常比暴力快)
         # Graham Scan 需要先找到最低点，但这里传入已按x排序的点，
         # Graham Scan 内部会重新找最低点并排序，所以没问题。
         return convex_hull_graham_scan(points_copy)

    # 2. 分割
    mid = n // 2
    left_points = points_copy[:mid]
    right_points = points_copy[mid:]

    # 3. 递归求解
    left_hull = convex_hull_divide_conquer(left_points)
    right_hull = convex_hull_divide_conquer(right_points)

    # 4. 合并 (Merge)
    return merge_hulls(left_hull, right_hull)

def merge_hulls(left_hull, right_hull):
    """合并左右两个凸包 (MergeHull 算法)"""
    n_left = len(left_hull)
    n_right = len(right_hull)

    # 处理空凸包的情况
    if n_left == 0: return right_hull
    if n_right == 0: return left_hull

    # 找到 left_hull 中 x 最大的点索引 (最右边的点)
    idx_left_max_x = 0
    for i in range(1, n_left):
        if left_hull[i].x > left_hull[idx_left_max_x].x:
            idx_left_max_x = i
        # 如果x坐标相同，选择y坐标较大的点（或较小的，保持一致即可）
        elif left_hull[i].x == left_hull[idx_left_max_x].x and left_hull[i].y > left_hull[idx_left_max_x].y:
             idx_left_max_x = i


    # 找到 right_hull 中 x 最小的点索引 (最左边的点)
    idx_right_min_x = 0
    for i in range(1, n_right):
        if right_hull[i].x < right_hull[idx_right_min_x].x:
            idx_right_min_x = i
        # 如果x坐标相同，选择y坐标较小的点（或较大的，与上面对应）
        elif right_hull[i].x == right_hull[idx_right_min_x].x and right_hull[i].y < right_hull[idx_right_min_x].y:
             idx_right_min_x = i

    # --- 计算上切线 (Upper Tangent) ---
    # 初始化切线为连接 left_hull 最右点和 right_hull 最左点的线段
    upper_l = idx_left_max_x
    upper_r = idx_right_min_x
    changed = True
    while changed:
        changed = False
        # 在 right_hull 中向上移动寻找更好的切点 (逆时针方向检查)
        # orientation(left_hull[upper_l], right_hull[upper_r], next_r) 应该是顺时针 (1) 或共线 (0)
        # 如果是逆时针 (2)，则 next_r 是更好的切点
        next_r = (upper_r + 1) % n_right
        while orientation(left_hull[upper_l], right_hull[upper_r], right_hull[next_r]) == 2: # 2 表示逆时针
            upper_r = next_r
            next_r = (upper_r + 1) % n_right
            changed = True

        # 在 left_hull 中向上移动寻找更好的切点 (顺时针方向检查)
        # orientation(right_hull[upper_r], left_hull[upper_l], prev_l) 应该是逆时针 (2) 或共线 (0)
        # 如果是顺时针 (1)，则 prev_l 是更好的切点
        prev_l = (upper_l - 1 + n_left) % n_left
        while orientation(right_hull[upper_r], left_hull[upper_l], left_hull[prev_l]) == 1: # 1 表示顺时针
            upper_l = prev_l
            prev_l = (upper_l - 1 + n_left) % n_left
            changed = True

    # --- 计算下切线 (Lower Tangent) ---
    # 初始化切线同上
    lower_l = idx_left_max_x
    lower_r = idx_right_min_x
    changed = True
    while changed:
        changed = False
        # 在 right_hull 中向下移动寻找更好的切点 (顺时针方向检查)
        # orientation(left_hull[lower_l], right_hull[lower_r], prev_r) 应该是逆时针 (2) 或共线 (0)
        # 如果是顺时针 (1)，则 prev_r 是更好的切点
        prev_r = (lower_r - 1 + n_right) % n_right
        while orientation(left_hull[lower_l], right_hull[lower_r], right_hull[prev_r]) == 1: # 1 表示顺时针
             lower_r = prev_r
             prev_r = (lower_r - 1 + n_right) % n_right
             changed = True

        # 在 left_hull 中向下移动寻找更好的切点 (逆时针方向检查)
        # orientation(right_hull[lower_r], left_hull[lower_l], next_l) 应该是顺时针 (1) 或共线 (0)
        # 如果是逆时针 (2)，则 next_l 是更好的切点
        next_l = (lower_l + 1) % n_left
        while orientation(right_hull[lower_r], left_hull[lower_l], left_hull[next_l]) == 2: # 2 表示逆时针
            lower_l = next_l
            next_l = (lower_l + 1) % n_left
            changed = True


    # --- 构建合并后的凸包 ---
    # 结果是按逆时针顺序排列的点
    merged_hull = []
    # 从上切线的左端点开始，沿 left_hull 逆时针走到下切线的左端点
    curr = upper_l
    merged_hull.append(left_hull[curr])
    # 当 left_hull 只有一个点时，curr == lower_l 会立即成立
    if n_left > 1:
        while curr != lower_l:
            curr = (curr + 1) % n_left # 沿逆时针方向移动
            merged_hull.append(left_hull[curr])

    # 从下切线的右端点开始，沿 right_hull 逆时针走到上切线的右端点
    curr = lower_r
    merged_hull.append(right_hull[curr])
    # 当 right_hull 只有一个点时，curr == upper_r 会立即成立
    if n_right > 1:
        while curr != upper_r:
            curr = (curr + 1) % n_right # 沿逆时针方向移动
            merged_hull.append(right_hull[curr])

    return merged_hull


# --- 4.4 对比三种凸包求解算法 ---

def compare_algorithms(sizes, num_runs=1):
    """
    对比不同凸包算法的性能。
    Args:
        sizes (list): 包含不同点集大小的列表, e.g., [1000, 2000, 3000]。
        num_runs (int): 对每个规模运行算法的次数，用于求平均时间。
    Returns:
        dict: 包含每个算法在不同规模下的平均运行时间的字典。
    """
    results = {
        "枚举法 (Brute Force)": [],
        "Graham Scan": [],
        "分治法 (Divide and Conquer)": []
    }
    algorithms = {
        "枚举法 (Brute Force)": convex_hull_brute_force,
        "Graham Scan": convex_hull_graham_scan,
        "分治法 (Divide and Conquer)": convex_hull_divide_conquer
    }

    print("开始性能测试...")
    for size in sizes:
        print(f"  测试点数: {size}")
        total_times = {name: 0.0 for name in algorithms}
        valid_runs = {name: num_runs for name in algorithms} # 记录有效运行次数

        for run in range(num_runs):
            # (1) 生成随机点集
            points = generate_random_points(size, 0, 100) # 在 (0,0)-(100,100) 内生成

            for name, func in algorithms.items():
                # 对于 Brute Force，如果点数过多，跳过以节省时间
                if name == "枚举法 (Brute Force)" and size > 300: # 降低阈值以更快看到结果
                    if total_times[name] != float('inf'): # 第一次遇到时标记
                        total_times[name] = float('inf')
                        valid_runs[name] = 0 # 无有效运行
                    continue # 跳过当前算法的本次运行

                if total_times[name] == float('inf'): continue # 如果已标记，跳过

                try:
                    start_time = time.perf_counter()
                    # 传入副本以防算法修改原始列表 (Graham Scan 会修改)
                    hull = func(points.copy())
                    end_time = time.perf_counter()
                    total_times[name] += (end_time - start_time)
                except Exception as e:
                    print(f"    错误发生在 {name} (点数 {size}, 运行 {run+1}): {e}")
                    total_times[name] = float('inf') # 标记为无效
                    valid_runs[name] = 0
                    break # 当前 size 的该算法后续运行也跳过

                # 可选：验证凸包结果是否一致 (用于调试)
                # print(f"    {name} Hull Size: {len(hull)}")

        # 计算平均时间
        for name in algorithms:
             if valid_runs[name] > 0:
                avg_time = total_times[name] / valid_runs[name]
                results[name].append(avg_time)
                print(f"    {name}: {avg_time:.6f} 秒 (平均 {valid_runs[name]} 次运行)")
             else:
                results[name].append(float('inf')) # 添加标记
                print(f"    {name}: 跳过或出错 (点数过多或运行时错误)")


    print("性能测试完成。")
    return results

def plot_performance(sizes, results):
    """绘制算法性能曲线"""
    plt.figure(figsize=(10, 6))

    for name, times in results.items():
        # 过滤掉未计算或出错的点 (inf)
        valid_indices = [i for i, t in enumerate(times) if t != float('inf')]
        if not valid_indices: continue # 如果没有有效数据点，则跳过绘制

        valid_sizes = [sizes[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]

        plt.plot(valid_sizes, valid_times, marker='o', linestyle='-', label=name)

    plt.xlabel("点数 (n)")
    plt.ylabel("平均运行时间 (秒)")
    plt.title("不同凸包算法性能对比")
    plt.legend()
    plt.grid(True)
    # 根据时间跨度决定是否使用对数刻度
    max_time = 0
    for times in results.values():
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            max_time = max(max_time, max(valid_times))

    if max_time > 10: # 如果最大时间超过10秒，用对数刻度可能更好
        plt.yscale('log')
        plt.ylabel("平均运行时间 (秒, 对数刻度)")

    plt.show()

# --- 新增：可视化算法结果 ---
def plot_hull_comparison(points, hulls, names):
    """
    绘制原始点集和不同算法计算出的凸包。
    Args:
        points (list): Point 对象的列表。
        hulls (list): 一个包含多个凸包的列表，每个凸包是 Point 对象的列表。
        names (list): 与 hulls 对应的算法名称列表。
    """
    plt.figure(figsize=(8, 8))
    # 绘制原始点
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    plt.scatter(x_coords, y_coords, c='blue', label='原始点', s=10, alpha=0.6) # s是点的大小, alpha是透明度

    colors = ['red', 'green', 'purple'] # 为不同算法的凸包指定颜色
    linestyles = ['-', '--', ':']      # 为不同算法的凸包指定线型

    for i, hull in enumerate(hulls):
        if not hull: # 如果某个算法返回空凸包，则跳过
            print(f"警告: 算法 '{names[i]}' 返回了空凸包，无法绘制。")
            continue
        # 将凸包闭合，以便绘制成多边形
        closed_hull = hull + [hull[0]]
        hull_x = [p.x for p in closed_hull]
        hull_y = [p.y for p in closed_hull]
        plt.plot(hull_x, hull_y, marker='o', markersize=5,
                 linestyle=linestyles[i % len(linestyles)],
                 color=colors[i % len(colors)],
                 label=f'{names[i]} 凸包 ({len(hull)}个顶点)') # 显示顶点数

    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("凸包算法结果可视化对比")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 保持 x 和 y 轴比例相等，使图形不变形
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # --- 示例运行与可视化 ---
    print("--- 示例运行与可视化 ---")
    # 使用一个稍大一点的随机点集进行可视化
    example_size = 50
    example_points = generate_random_points(example_size, 0, 100)
    print(f"生成 {example_size} 个随机点用于可视化...")
    # print("输入点:", example_points) # 点太多时不打印

    hulls_to_plot = []
    hull_names = []

    print("\n计算枚举法凸包...")
    try:
        hull_bf = convex_hull_brute_force(example_points)
        hulls_to_plot.append(hull_bf)
        hull_names.append("枚举法")
        # print("枚举法结果:", hull_bf)
    except Exception as e:
        print(f"枚举法计算出错: {e}")

    print("\n计算 Graham Scan 凸包...")
    try:
        # Graham Scan 可能修改输入列表，传入副本
        hull_gs = convex_hull_graham_scan(example_points.copy())
        hulls_to_plot.append(hull_gs)
        hull_names.append("Graham Scan")
        # print("Graham Scan 结果:", hull_gs)
    except Exception as e:
        print(f"Graham Scan 计算出错: {e}")


    print("\n计算分治法凸包...")
    try:
        hull_dc = convex_hull_divide_conquer(example_points)
        hulls_to_plot.append(hull_dc)
        hull_names.append("分治法")
        # print("分治法结果:", hull_dc)
    except Exception as e:
        print(f"分治法计算出错: {e}")

    # 调用新的绘图函数进行可视化
    if hulls_to_plot:
        print("\n绘制算法结果对比图...")
        plot_hull_comparison(example_points, hulls_to_plot, hull_names)
    else:
        print("没有成功的凸包结果可供绘制。")


    # --- 性能对比测试 ---
    print("\n--- 性能对比 ---")
    # (2) 定义不同的数据集合大小
    point_sizes = [50, 100, 200, 300, 500, 1000, 2000, 3000] # 调整规模以便更快看到结果
    num_runs_per_size = 2 # 每个规模运行次数

    # (3) 运行算法并记录时间
    performance_data = compare_algorithms(point_sizes, num_runs_per_size)

    # (4) 绘制性能曲线
    print("\n绘制性能对比曲线...")
    plot_performance(point_sizes, performance_data)

    print("\n程序执行完毕。")

