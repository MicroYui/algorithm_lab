import numpy as np
import matplotlib.pyplot as plt
import time
from math import atan2


# 点类，用于表示平面上的点
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"({self.x}, {self.y})"


# 计算两个向量的叉积
def cross_product(p1, p2, p3):
    """计算向量p1p2和p1p3的叉积"""
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)


# 判断点a是否在三角形bcd内部
def is_inside_triangle(a, b, c, d):
    """判断点a是否在三角形bcd内部"""
    # 利用叉积判断点a是否在三条边的同一侧
    cross1 = cross_product(b, c, a)
    cross2 = cross_product(c, d, a)
    cross3 = cross_product(d, b, a)

    # 如果三个叉积的符号相同，说明点a在三角形内部
    if (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0):
        return True
    return False


# 计算两个点之间的欧氏距离
def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


# 计算极角
def polar_angle(p, reference):
    """计算点p相对于参考点reference的极角"""
    return atan2(p.y - reference.y, p.x - reference.x)


# 判断三点的方向
def orientation(p, q, r):
    """计算p->q->r的转向"""
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0  # 共线
    return 1 if val > 0 else 2  # 顺时针或逆时针


# 随机生成点集合
def generate_points(n, size=100):
    """随机生成n个点，范围在0到size之间"""
    points = []
    for _ in range(n):
        x = np.random.uniform(0, size)
        y = np.random.uniform(0, size)
        points.append(Point(x, y))
    return points


# ---------------------- 枚举法求凸包 ----------------------
def convex_hull_enumeration(points):
    """
    基于枚举方法的凸包求解算法
    思路：对于每一条可能的边，判断是否所有其他点都在这条边的同一侧
    """
    n = len(points)
    if n <= 3:
        return points  # 点数小于等于3，所有点都是凸包的顶点

    hull = []

    # 枚举所有可能的边
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # 当前考虑的边：points[i] -> points[j]
            is_edge = True

            # 记录第一个不是i和j的点在边的哪一侧
            side = None

            for k in range(n):
                if k == i or k == j:
                    continue

                # 计算叉积，判断点k在边的哪一侧
                cross = cross_product(points[i], points[j], points[k])

                if side is None:
                    # 记录第一个点的位置
                    side = cross > 0
                elif (cross > 0) != side:
                    # 如果有点不在同一侧，则不是凸包的边
                    is_edge = False
                    break

            if is_edge:
                # 如果是凸包的边，则保存这两个点
                if points[i] not in hull:
                    hull.append(points[i])
                if points[j] not in hull:
                    hull.append(points[j])

    # 对凸包顶点按照极角排序
    if hull:
        center = Point(sum(p.x for p in hull) / len(hull), sum(p.y for p in hull) / len(hull))
        hull.sort(key=lambda p: atan2(p.y - center.y, p.x - center.x))

    return hull


# ---------------------- Graham-Scan求凸包 ----------------------
def convex_hull_graham_scan(points):
    """
    基于Graham-Scan的凸包求解算法
    步骤：
    1. 找到最低且最左的点
    2. 根据极角排序其他点
    3. 沿着排序后的点依次进行，保持逆时针转向
    """
    if len(points) <= 3:
        return points

    # 找到最低且最左的点
    lowest = min(points, key=lambda p: (p.y, p.x))

    # 按照极角排序
    sorted_points = sorted(points, key=lambda p: (atan2(p.y - lowest.y, p.x - lowest.x),
                                                  distance(p, lowest)))

    # 构建凸包
    hull = []
    for point in sorted_points:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], point) != 2:
            hull.pop()
        hull.append(point)

    return hull


# ---------------------- 分治法求凸包（新版本）----------------------
def convex_hull_divide_and_conquer(points):
    """
    基于分治思想的凸包求解算法（新版本）
    步骤：
    1. 分割点集为左右两部分
    2. 递归求解左右两部分的凸包
    3. 合并左右两个凸包
    """
    # 基本情况处理
    if len(points) <= 3:
        if len(points) == 3:
            # 检查三点是否共线
            if orientation(points[0], points[1], points[2]) == 0:
                # 返回两个端点
                return sorted(points, key=lambda p: (p.x, p.y))[:2]
        return points

    # 按x坐标排序
    sorted_points = sorted(points, key=lambda p: p.x)

    # 分割点集
    mid = len(sorted_points) // 2
    left_points = sorted_points[:mid]
    right_points = sorted_points[mid:]

    # 递归求解左右两部分的凸包
    left_hull = convex_hull_divide_and_conquer(left_points)
    right_hull = convex_hull_divide_and_conquer(right_points)

    # 合并左右凸包
    return merge_hulls(left_hull, right_hull)


def merge_hulls(left_hull, right_hull):
    """
    按照新思路合并两个凸包
    1. 找左侧凸包中的一个内点p
    2. 在右侧凸包中找与p的极角最大和最小的顶点u和v
    3. 构造三个序列并合并
    4. 在合并的序列上应用Graham-Scan
    """
    # 特殊情况处理
    if not left_hull:
        return right_hull
    if not right_hull:
        return left_hull

    # 1. 找左侧凸包中的一个内点p
    # 这里简单地取左凸包各点的平均位置作为内点
    p = Point(sum(point.x for point in left_hull) / len(left_hull),
              sum(point.y for point in left_hull) / len(left_hull))

    # 2. 在右侧凸包中找与p的极角最大和最小的顶点u和v
    angles = [(i, polar_angle(point, p)) for i, point in enumerate(right_hull)]

    min_angle_idx = min(angles, key=lambda x: x[1])[0]
    max_angle_idx = max(angles, key=lambda x: x[1])[0]

    u = right_hull[min_angle_idx]  # 极角最小的点
    v = right_hull[max_angle_idx]  # 极角最大的点

    # 3. 构造三个序列并合并
    # (1) 按逆时针方向排列的左凸包的所有顶点
    left_center = Point(sum(p.x for p in left_hull) / len(left_hull),
                        sum(p.y for p in left_hull) / len(left_hull))
    sorted_left = sorted(left_hull, key=lambda point: polar_angle(point, left_center))

    # (2) 按逆时针方向排列的右凸包从u到v的顶点
    sorted_right_u_to_v = []
    i = min_angle_idx
    while True:
        sorted_right_u_to_v.append(right_hull[i])
        if i == max_angle_idx:
            break
        i = (i + 1) % len(right_hull)

    # (3) 按顺时针方向排列的右凸包从u到v的顶点
    sorted_right_v_to_u = []
    i = max_angle_idx
    while True:
        sorted_right_v_to_u.append(right_hull[i])
        if i == min_angle_idx:
            break
        i = (i - 1) % len(right_hull)

    # 4. 合并上述三个序列
    merged_points = sorted_left + sorted_right_u_to_v + sorted_right_v_to_u

    # 5. 在合并的序列上应用Graham-Scan
    return convex_hull_graham_scan(merged_points)


# ---------------------- 算法性能测试与比较 ----------------------
def compare_algorithms(sizes=[1000, 2000, 3000, 4000, 5000]):
    """
    比较三种算法在不同规模数据下的性能
    """
    results = {'枚举法': [], 'Graham-Scan': [], '分治法': []}

    for size in sizes:
        print(f"生成{size}个点的数据集...")
        points = generate_points(size)

        # 测试枚举法（仅对小规模数据）
        # if size <= 2000:  # 枚举法对大数据集效率较低，这里限制一下
        #     print(f"运行枚举法...")
        #     start_time = time.time()
        #     convex_hull_enumeration(points)
        #     end_time = time.time()
        #     results['枚举法'].append((size, end_time - start_time))
        # else:
        #     results['枚举法'].append((size, None))

        print(f"运行枚举法...")
        start_time = time.time()
        convex_hull_enumeration(points)
        end_time = time.time()
        results['枚举法'].append((size, end_time - start_time))

        # 测试Graham-Scan
        print(f"运行Graham-Scan...")
        start_time = time.time()
        convex_hull_graham_scan(points)
        end_time = time.time()
        results['Graham-Scan'].append((size, end_time - start_time))

        # 测试分治法
        print(f"运行分治法...")
        start_time = time.time()
        convex_hull_divide_and_conquer(points)
        end_time = time.time()
        results['分治法'].append((size, end_time - start_time))

    return results


# ---------------------- 绘制算法性能曲线 ----------------------
def plot_performance(results):
    """
    绘制算法性能曲线
    """
    plt.figure(figsize=(10, 6))

    # 设置中文字体，解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    for method, data in results.items():
        sizes = [item[0] for item in data]
        times = [item[1] for item in data]

        # 过滤掉None值
        valid_sizes = [sizes[i] for i in range(len(sizes)) if times[i] is not None]
        valid_times = [times[i] for i in range(len(times)) if times[i] is not None]

        if valid_sizes:
            plt.plot(valid_sizes, valid_times, 'o-', label=method)

    plt.xlabel('数据集大小')
    plt.ylabel('运行时间(秒)')
    plt.title('凸包算法性能比较')
    plt.legend()
    plt.grid(True)

    # 保存图像到当前目录
    try:
        plt.savefig('凸包算法性能比较.png')
        print("性能比较图像已保存为：凸包算法性能比较.png")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 尝试显示图像，但如果环境不支持则忽略错误
    try:
        plt.show()
    except Exception as e:
        print(f"显示图像时出错: {e}")
        print("图像已保存，但无法在当前环境中显示")


# ---------------------- 可视化凸包 ----------------------
def visualize_convex_hull(points, hull, title):
    """
    可视化点集和凸包
    """
    plt.figure(figsize=(8, 8))

    # 设置中文字体，解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制所有点
    plt.scatter([p.x for p in points], [p.y for p in points], color='blue', label='点集')

    # 绘制凸包
    hull_x = [p.x for p in hull]
    hull_y = [p.y for p in hull]
    hull_x.append(hull[0].x)  # 闭合凸包
    hull_y.append(hull[0].y)
    plt.plot(hull_x, hull_y, 'r-', linewidth=2, label='凸包')

    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图像，使用不含特殊字符的文件名
    safe_title = ''.join(c for c in title if c.isalnum() or c in [' ', '_', '-'])
    filename = f'{safe_title}.png'
    try:
        plt.savefig(filename)
        print(f"凸包图像已保存为：{filename}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 尝试显示图像，但如果环境不支持则忽略错误
    try:
        plt.show()
    except Exception as e:
        print(f"显示图像时出错: {e}")
        print("图像已保存，但无法在当前环境中显示")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    try:
        # 设置随机种子，确保结果可重现
        np.random.seed(42)

        print("开始凸包算法实验...")

        # 生成一个小规模的点集用于测试和可视化
        print("生成测试点集...")
        test_points = generate_points(30)

        # 使用三种算法求解凸包
        print("使用枚举法求解凸包...")
        hull_enum = convex_hull_enumeration(test_points)
        print("使用Graham-Scan求解凸包...")
        hull_graham = convex_hull_graham_scan(test_points)
        print("使用分治法求解凸包...")
        hull_dc = convex_hull_divide_and_conquer(test_points)

        # 可视化结果
        print("\n开始可视化结果...")
        visualize_convex_hull(test_points, hull_enum, "枚举法求凸包")
        visualize_convex_hull(test_points, hull_graham, "Graham-Scan求凸包")
        visualize_convex_hull(test_points, hull_dc, "分治法求凸包")

        # 比较算法性能
        print("\n开始比较算法性能...")
        sizes = [1000, 2000, 3000, 4000, 5000]
        results = compare_algorithms(sizes)

        # 绘制性能曲线
        print("\n绘制性能曲线...")
        plot_performance(results)

        print("\n实验完成！结果已保存为图片文件。")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()