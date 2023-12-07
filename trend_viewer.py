import matplotlib.pyplot as plt
import pandas as pd

# 读取面板数据
loca=
variable = '人均地区生产总值(元)'
idlist = ['Guangdong']
idtag = '地区'
yeartag = '年份'
dashline = 2008

# 获取所有唯一的省份


panel_data = pd.read_excel(loca)
provinces = panel_data[idtag].unique()

# 创建图表
plt.figure(figsize=(6,6))  # 可根据需要调整图表大小

# 绘制其他省份的曲线
for province in provinces:
    if province not in idlist:
        province_data = panel_data[panel_data[idtag] == province]
        plt.plot(province_data[yeartag], province_data[variable], color='gray')

# 绘制特殊省份的曲线
for province in idlist:
    province_data = panel_data[panel_data[idtag] == province]
    plt.plot(province_data[yeartag], province_data[variable], label='{}'.format(province), linewidth=2)

# 添加虚线
plt.axvline(x=dashline, color='r', linestyle='--')
plt.xlim(2005, 2019)


# 设置图表标题和轴标签
plt.xlabel('Year')
plt.ylabel(variable)

# 添加图例
plt.legend()

# 显示图表
plt.show()
