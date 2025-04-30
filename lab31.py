"""
实验三：近似算法 - 集合覆盖问题
- 实现基于贪心策略的近似算法
- 实现基于线性规划的近似算法（舍入法）
"""

import random
import time as time_module  # 重命名避免与变量冲突
import matplotlib.pyplot as plt
import numpy as np
from pulp import *
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class SetCoverProblem:
    def __init__(self, X_size, F_size):
        """
        初始化集合覆盖问题
        :param X_size: 有限集X的大小
        :param F_size: 子集族F的大小
        """
        self.X_size = X_size
        self.F_size = F_size
        self.X = set(range(X_size))  # 有限集X
        self.F = []  # X的子集族F

        # 生成子集族
        self.generate_subsets()

    def generate_subsets(self):
        """
        生成子集族F，确保存在一个可行解
        """
        X_list = list(self.X)

        # 第一步：生成可行解集合
        # 先随机选择20个点放入S0中
        S0 = set(random.sample(X_list, min(20, self.X_size)))
        self.F.append(S0)

        covered = S0.copy()  # 已覆盖的元素
        remaining = self.X.difference(covered)  # 剩余未覆盖的元素

        # 继续生成集合直到所有元素都被覆盖或者达到一定数量的集合
        i = 1
        while remaining and i < self.F_size // 5:
            n = random.randint(1, 20)  # 新集合大小
            x = random.randint(1, min(n, len(remaining)))  # 从remaining中选择x个点

            S_new = set(random.sample(list(remaining), x))
            covered_sample = random.sample(list(covered), min(n-x, len(covered)))
            S_new.update(covered_sample)

            self.F.append(S_new)
            covered.update(S_new)
            remaining = self.X.difference(covered)

            i += 1

            # 如果剩余元素很少，直接把它们作为最后一个集合
            if len(remaining) < 20:
                if remaining:
                    self.F.append(remaining)
                    covered.update(remaining)
                    remaining = set()

        # 第二步：继续生成其余集合以达到F_size
        while len(self.F) < self.F_size:
            set_size = random.randint(1, min(20, self.X_size))
            new_set = set(random.sample(X_list, set_size))
            self.F.append(new_set)

    def greedy_approximation(self):
        """
        基于贪心策略的近似算法
        按照标准伪代码实现：
        1. U←X; /* U是X中尚未被覆盖的元素集 */
        2. C←∅;
        3. While U≠∅ Do
        4.     Select S∈F 使得|S∩U|最大;
        5.     U← U-S;
        6.     C← C∪{S}; /* 构造X的覆盖 */
        7. Return C.

        返回：一个近似最小的集合覆盖C
        """
        start_time = time_module.time()

        # 1. U←X; /* U是X中尚未被覆盖的元素集 */
        U = self.X.copy()
        # 2. C←∅;
        C = []

        # 3. While U≠∅ Do
        while U:
            # 4. Select S∈F 使得|S∩U|最大;
            best_set = None
            max_covered = -1

            for i, S in enumerate(self.F):
                covered = len(S.intersection(U))
                if covered > max_covered:
                    max_covered = covered
                    best_set = i

            # 如果找不到能覆盖更多元素的集合，则退出
            if max_covered == 0:
                break

            # 5. U← U-S;
            U = U.difference(self.F[best_set])
            # 6. C← C∪{S}; /* 构造X的覆盖 */
            C.append(best_set)

        # 7. Return C.
        end_time = time_module.time()
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒

        # 检查是否完全覆盖
        covered = set()
        for idx in C:
            covered.update(self.F[idx])

        is_complete = covered == self.X

        return {
            'indices': C,  # 选择的集合索引
            'size': len(C),  # 覆盖大小
            'time': execution_time,  # 执行时间（毫秒）
            'complete': is_complete  # 是否完全覆盖
        }

    def lp_approximation(self):
        """
        基于线性规划的近似算法（舍入法）
        返回：一个近似最小的集合覆盖C
        """
        start_time = time_module.time()

        # 创建问题
        prob = LpProblem("SetCover", LpMinimize)

        # 创建变量
        x = {}
        for i in range(len(self.F)):
            x[i] = LpVariable(f"x_{i}", 0, 1)  # x_i 是一个介于0和1之间的变量

        # 目标函数：最小化选择的集合数量
        prob += lpSum([x[i] for i in range(len(self.F))])

        # 约束条件：每个元素至少被一个选定的集合覆盖
        for e in self.X:
            prob += lpSum([x[i] for i in range(len(self.F)) if e in self.F[i]]) >= 1

        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=False))

        # 使用舍入法：选择所有x_i >= 0.5的集合
        C = []
        for i in range(len(self.F)):
            if value(x[i]) >= 0.5:
                C.append(i)

        # 检查是否完全覆盖，如果没有则添加必要的集合
        covered = set()
        for idx in C:
            covered.update(self.F[idx])

        # 如果还有未覆盖的元素，使用贪心方法添加额外的集合
        uncovered = self.X.difference(covered)
        while uncovered:
            best_set = None
            max_covered = -1

            for i, S in enumerate(self.F):
                if i in C:  # 跳过已选集合
                    continue
                covered_new = len(S.intersection(uncovered))
                if covered_new > max_covered:
                    max_covered = covered_new
                    best_set = i

            if max_covered == 0:
                break

            C.append(best_set)
            uncovered = uncovered.difference(self.F[best_set])

        end_time = time_module.time()
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒

        # 再次检查是否完全覆盖
        covered = set()
        for idx in C:
            covered.update(self.F[idx])

        is_complete = covered == self.X

        return {
            'indices': C,  # 选择的集合索引
            'size': len(C),  # 覆盖大小
            'time': execution_time,  # 执行时间（毫秒）
            'complete': is_complete  # 是否完全覆盖
        }

def run_experiment(sizes):
    """
    运行实验并比较两种算法
    :param sizes: 要测试的问题规模列表
    """
    results = {
        'size': sizes,
        'greedy': {
            'time': [],
            'cover_size': []
        },
        'lp': {
            'time': [],
            'cover_size': []
        }
    }

    for size in sizes:
        print(f"运行规模为 {size} 的实验...")

        # 创建问题实例
        problem = SetCoverProblem(size, size)

        # 运行贪心算法
        greedy_result = problem.greedy_approximation()
        print(f"  贪心算法 - 覆盖大小：{greedy_result['size']}, 时间：{greedy_result['time']:.2f}毫秒, 完全覆盖：{greedy_result['complete']}")
        print(f"  选择的集合索引：{greedy_result['indices'][:10]}..." if len(greedy_result['indices']) > 10 else f"  选择的集合索引：{greedy_result['indices']}")

        # 对于大规模问题，LP算法可能会非常慢，如果规模大于1000可能需要很长时间
        if size <= 5000:
            # 运行线性规划算法
            lp_result = problem.lp_approximation()
            print(f"  线性规划 - 覆盖大小：{lp_result['size']}, 时间：{lp_result['time']:.2f}毫秒, 完全覆盖：{lp_result['complete']}")
            print(f"  选择的集合索引：{lp_result['indices'][:10]}..." if len(lp_result['indices']) > 10 else f"  选择的集合索引：{lp_result['indices']}")
        else:
            print("  线性规划 - 由于问题规模太大(5000)而跳过，可能需要很长时间运行")
            lp_result = {'size': None, 'time': None, 'complete': None}

        # 记录结果
        results['greedy']['time'].append(greedy_result['time'])
        results['greedy']['cover_size'].append(greedy_result['size'])

        if lp_result['time'] is not None:
            results['lp']['time'].append(lp_result['time'])
            results['lp']['cover_size'].append(lp_result['size'])

    return results

def plot_results(results):
    """
    绘制算法性能曲线
    :param results: 实验结果
    """
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制时间性能曲线
    ax1.plot(results['size'], results['greedy']['time'], 'o-', label='贪心算法')
    if len(results['lp']['time']) > 0:
        ax1.plot(results['size'][:len(results['lp']['time'])], results['lp']['time'], 's-', label='线性规划')
    ax1.set_xlabel('问题规模 |X| = |F|')
    ax1.set_ylabel('执行时间 (毫秒)')
    ax1.set_title('算法执行时间比较')
    ax1.legend()
    ax1.grid(True)

    # 绘制覆盖大小曲线
    ax2.plot(results['size'], results['greedy']['cover_size'], 'o-', label='贪心算法')
    if len(results['lp']['cover_size']) > 0:
        ax2.plot(results['size'][:len(results['lp']['cover_size'])], results['lp']['cover_size'], 's-', label='线性规划')
    ax2.set_xlabel('问题规模 |X| = |F|')
    ax2.set_ylabel('覆盖大小 |C|')
    ax2.set_title('算法覆盖大小比较')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('set_cover_results.png')

    # 打印实验结果表格
    print("\n=== 实验结果汇总 ===")
    print("问题规模\t贪心时间(ms)\t贪心覆盖大小\t线性规划时间(ms)\t线性规划覆盖大小")
    for i, size in enumerate(results['size']):
        greedy_time = results['greedy']['time'][i]
        greedy_size = results['greedy']['cover_size'][i]
        lp_time = results['lp']['time'][i] if i < len(results['lp']['time']) else "N/A"
        lp_size = results['lp']['cover_size'][i] if i < len(results['lp']['cover_size']) else "N/A"
        print(f"{size}\t{greedy_time:.2f}\t{greedy_size}\t{lp_time}\t{lp_size}")

    plt.show()

def main():
    # 设置随机种子，使结果可重现
    random.seed(42)

    # 测试小规模实例，确保算法正确性
    test_problem = SetCoverProblem(100, 100)

    print("=== 小规模测试 (|X|=|F|=100) ===")
    greedy_result = test_problem.greedy_approximation()
    print(f"贪心算法 - 覆盖大小：{greedy_result['size']}, 时间：{greedy_result['time']:.2f}毫秒")
    print(f"选择的集合索引：{greedy_result['indices']}")

    lp_result = test_problem.lp_approximation()
    print(f"线性规划 - 覆盖大小：{lp_result['size']}, 时间：{lp_result['time']:.2f}毫秒")
    print(f"选择的集合索引：{lp_result['indices']}")

    # 运行不同规模的实验
    print("\n=== 不同规模的实验 ===")
    # 使用实验要求的规模：100, 1000, 5000
    sizes = [100, 1000, 5000]
    results = run_experiment(sizes)

    # 绘制结果
    plot_results(results)

if __name__ == "__main__":
    main()