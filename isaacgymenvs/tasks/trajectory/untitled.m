% 1. 导入CSV文件数据
data = readtable('helix.csv');  % 替换 'your_file.csv' 为实际文件名

% 2. 提取x、y和z坐标
x = data(:, 1);
y = data(:, 2);
z = data(:, 3);

% 3. 绘制三维图像
figure;
plot3(x, y, z, 'o-', 'LineWidth', 2);
grid on;
xlabel('X轴');
ylabel('Y轴');
zlabel('Z轴');
title('三维轨迹图');

% 可以根据需要添加其他绘图设置和样式
