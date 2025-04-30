import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


# 4.1 按照算法导论中给出的伪代码实现快速排序
def rand_partition(arr, p, r):
    """
    随机选择一个元素作为基准，并进行划分
    参数:
        arr: 待排序数组
        p: 起始索引
        r: 结束索引
    返回值:
        基准元素的最终位置
    """
    i = random.randint(p, r)  # 随机选择一个索引
    arr[i], arr[p] = arr[p], arr[i]  # 将随机选择的元素与首元素交换
    x = arr[p]  # 记录基准元素
    i = p - 1

    for j in range(p, r):
        if arr[j] <= x:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 将小于等于基准的元素放到左边

    arr[i + 1], arr[r] = arr[r], arr[i + 1]  # 将基准元素放到最终位置
    return i + 1


def quick_sort(arr, p, r):
    """
    实现快速排序算法
    参数:
        arr: 待排序数组
        p: 起始索引
        r: 结束索引
    """
    if p < r:
        q = rand_partition(arr, p, r)  # 进行划分
        quick_sort(arr, p, q - 1)  # 对基准左边的子数组进行排序
        quick_sort(arr, q + 1, r)  # 对基准右边的子数组进行排序


# 优化的快速排序版本 - 处理重复元素的情况
def optimized_partition(arr, p, r):
    """
    优化的划分方法，处理重复元素
    参数:
        arr: 待排序数组
        p: 起始索引
        r: 结束索引
    返回值:
        元组 (i, j)，i是第一个等于基准元素的位置，j是最后一个等于基准元素的位置
    """
    # 三数取中选择基准
    mid = (p + r) // 2
    if arr[p] > arr[mid]:
        arr[p], arr[mid] = arr[mid], arr[p]
    if arr[p] > arr[r]:
        arr[p], arr[r] = arr[r], arr[p]
    if arr[mid] > arr[r]:
        arr[mid], arr[r] = arr[r], arr[mid]

    # 使用中间元素作为基准
    arr[p], arr[mid] = arr[mid], arr[p]
    pivot = arr[p]

    # 三路划分: <pivot | =pivot | >pivot
    i = p  # 小于基准的最后一个位置
    j = p  # 当前考虑的元素
    k = r  # 大于基准的第一个位置

    while j < k:
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1
        elif arr[j] > pivot:
            arr[j], arr[k] = arr[k], arr[j]
            k -= 1
        else:
            j += 1

    return i, j - 1


def optimized_quick_sort(arr, p, r):
    """
    优化的快速排序，针对包含大量重复元素的数组
    参数:
        arr: 待排序数组
        p: 起始索引
        r: 结束索引
    """
    if p < r:
        # 对于小规模数组使用插入排序
        if r - p < 16:
            insertion_sort(arr, p, r)
            return

        lt, gt = optimized_partition(arr, p, r)
        optimized_quick_sort(arr, p, lt - 1)  # 排序小于基准的部分
        optimized_quick_sort(arr, gt + 1, r)  # 排序大于基准的部分
        # 等于基准的部分不需要排序


def insertion_sort(arr, p, r):
    """
    插入排序，用于小规模子数组
    参数:
        arr: 待排序数组
        p: 起始索引
        r: 结束索引
    """
    for i in range(p + 1, r + 1):
        key = arr[i]
        j = i - 1
        while j >= p and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# 生成测试数据
def generate_test_data(size, duplicate_percentage):
    """
    生成测试数据集
    参数:
        size: 数据集大小
        duplicate_percentage: 重复元素的百分比
    返回:
        生成的数据数组
    """
    # 计算不同元素的数量
    unique_count = int(size * (1 - duplicate_percentage / 100))
    if unique_count <= 0:
        unique_count = 1

    # 生成唯一元素
    unique_elements = list(range(unique_count))

    # 生成数组
    arr = []
    for _ in range(size):
        arr.append(random.choice(unique_elements))

    # 打乱数组
    random.shuffle(arr)
    return arr


# 测试算法在不同输入下的表现
def run_experiment_4_2():
    """
    运行实验4.2: 测试算法在不同输入下的表现
    """
    print("====== 实验4.2: 测试算法在不同输入下的表现 ======")

    # 创建结果记录
    n = 10 ** 5  # 使用较小的数据规模进行测试，可以根据实际情况调整
    results = {
        'duplicate_percentages': [],
        'standard_quicksort_times': [],
        'python_sort_times': []
    }

    # 对不同重复率的数据集进行测试
    for i in range(11):
        duplicate_percentage = 10 * i
        results['duplicate_percentages'].append(duplicate_percentage)

        # 生成测试数据
        test_data = generate_test_data(n, duplicate_percentage)
        test_data_copy = test_data.copy()

        # 测试标准快速排序
        start_time = time.time()
        # 对于大规模数据，我们可能需要增加递归深度限制
        import sys
        sys.setrecursionlimit(10000)  # 增加递归深度限制
        try:
            quick_sort(test_data, 0, len(test_data) - 1)
            end_time = time.time()
            standard_time = end_time - start_time
            print(f"标准快速排序 - 重复率 {duplicate_percentage}%: {standard_time:.6f} 秒")
            results['standard_quicksort_times'].append(standard_time)
        except RecursionError:
            print(f"标准快速排序 - 重复率 {duplicate_percentage}%: 递归深度超限")
            results['standard_quicksort_times'].append(None)

        # 测试Python内置排序
        start_time = time.time()
        test_data_copy.sort()
        end_time = time.time()
        python_time = end_time - start_time
        print(f"Python内置排序 - 重复率 {duplicate_percentage}%: {python_time:.6f} 秒")
        results['python_sort_times'].append(python_time)

    return results


# 4.3 思考并解决问题
def run_experiment_4_3():
    """
    运行实验4.3: 解决问题，优化快速排序算法
    """
    print("\n====== 实验4.3: 优化快速排序算法 ======")

    # 创建结果记录
    n = 10 ** 5  # 使用较小的数据规模进行测试，可以根据实际情况调整
    results = {
        'duplicate_percentages': [],
        'standard_quicksort_times': [],
        'optimized_quicksort_times': [],
        'python_sort_times': []
    }

    # 对不同重复率的数据集进行测试
    for i in range(11):
        duplicate_percentage = 10 * i
        results['duplicate_percentages'].append(duplicate_percentage)

        # 生成测试数据
        test_data = generate_test_data(n, duplicate_percentage)
        test_data_copy1 = test_data.copy()
        test_data_copy2 = test_data.copy()

        # 测试标准快速排序
        start_time = time.time()
        try:
            quick_sort(test_data, 0, len(test_data) - 1)
            end_time = time.time()
            standard_time = end_time - start_time
            print(f"标准快速排序 - 重复率 {duplicate_percentage}%: {standard_time:.6f} 秒")
            results['standard_quicksort_times'].append(standard_time)
        except RecursionError:
            print(f"标准快速排序 - 重复率 {duplicate_percentage}%: 递归深度超限")
            results['standard_quicksort_times'].append(None)

        # 测试优化的快速排序
        start_time = time.time()
        optimized_quick_sort(test_data_copy1, 0, len(test_data_copy1) - 1)
        end_time = time.time()
        optimized_time = end_time - start_time
        print(f"优化的快速排序 - 重复率 {duplicate_percentage}%: {optimized_time:.6f} 秒")
        results['optimized_quicksort_times'].append(optimized_time)

        # 测试Python内置排序
        start_time = time.time()
        test_data_copy2.sort()
        end_time = time.time()
        python_time = end_time - start_time
        print(f"Python内置排序 - 重复率 {duplicate_percentage}%: {python_time:.6f} 秒")
        results['python_sort_times'].append(python_time)

    return results


# 绘制实验结果图表
def plot_results(results, title, filename):
    """
    绘制实验结果图表
    参数:
        results: 实验结果数据
        title: 图表标题
        filename: 保存文件名
    """
    plt.figure(figsize=(12, 8))
    duplicate_percentages = results['duplicate_percentages']

    # 绘制标准快速排序的时间
    if 'standard_quicksort_times' in results:
        # 过滤掉None值
        valid_indices = [i for i, x in enumerate(results['standard_quicksort_times']) if x is not None]
        if valid_indices:
            valid_percentages = [duplicate_percentages[i] for i in valid_indices]
            valid_times = [results['standard_quicksort_times'][i] for i in valid_indices]
            plt.plot(valid_percentages, valid_times, 'o-', label='标准快速排序')

    # 绘制优化的快速排序的时间
    if 'optimized_quicksort_times' in results:
        plt.plot(duplicate_percentages, results['optimized_quicksort_times'], 's-', label='优化的快速排序')

    # 绘制Python内置排序的时间
    if 'python_sort_times' in results:
        plt.plot(duplicate_percentages, results['python_sort_times'], '^-', label='Python内置排序')

    plt.xlabel('重复元素百分比 (%)')
    plt.ylabel('运行时间 (秒)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# 验证排序算法的正确性
def verify_algorithm_correctness():
    """
    验证排序算法的正确性
    """
    print("\n====== 验证排序算法的正确性 ======")

    # 生成小规模测试数据
    test_data = [5, 3, 8, 4, 2, 1, 9, 7, 6]
    print(f"原始数据: {test_data}")

    # 测试标准快速排序
    test_data_copy = test_data.copy()
    quick_sort(test_data_copy, 0, len(test_data_copy) - 1)
    print(f"标准快速排序结果: {test_data_copy}")
    print(f"标准快速排序正确性: {test_data_copy == sorted(test_data)}")

    # 测试优化的快速排序
    test_data_copy = test_data.copy()
    optimized_quick_sort(test_data_copy, 0, len(test_data_copy) - 1)
    print(f"优化的快速排序结果: {test_data_copy}")
    print(f"优化的快速排序正确性: {test_data_copy == sorted(test_data)}")

    # 测试带有重复元素的数据
    test_data = [5, 3, 5, 8, 3, 4, 2, 1, 9, 7, 6, 5]
    print(f"\n带重复元素的原始数据: {test_data}")

    # 测试标准快速排序
    test_data_copy = test_data.copy()
    quick_sort(test_data_copy, 0, len(test_data_copy) - 1)
    print(f"标准快速排序结果: {test_data_copy}")
    print(f"标准快速排序正确性: {test_data_copy == sorted(test_data)}")

    # 测试优化的快速排序
    test_data_copy = test_data.copy()
    optimized_quick_sort(test_data_copy, 0, len(test_data_copy) - 1)
    print(f"优化的快速排序结果: {test_data_copy}")
    print(f"优化的快速排序正确性: {test_data_copy == sorted(test_data)}")


# 主函数
def main():
    """
    主函数，运行所有实验
    """
    # 验证算法正确性
    verify_algorithm_correctness()

    # 运行实验4.2
    results_4_2 = run_experiment_4_2()
    plot_results(results_4_2, '不同重复率下快速排序与Python内置排序的性能比较', 'experiment_4_2.png')

    # 运行实验4.3
    results_4_3 = run_experiment_4_3()
    plot_results(results_4_3, '优化前后快速排序与Python内置排序的性能比较', 'experiment_4_3.png')


if __name__ == "__main__":
    main()