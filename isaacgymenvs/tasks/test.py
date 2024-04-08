import numpy as np

# 创建一个示例数组
array = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])

# 将大于1的元素修改为100
array[array > 1] = 100

print(array)
