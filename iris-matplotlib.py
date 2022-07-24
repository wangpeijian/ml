import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
# plt.title('花瓣宽度分布', fontproperties=cnFont)

cnFont = matplotlib.font_manager.FontProperties(fname='./font/SimHei.ttf')
mpl.rcParams['font.sans-serif'] = ['SimHei']

# pandas操作数据
df = pd.read_csv("./dataset/iris.data", names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类型'])
print(df.describe(percentiles=[.20, .40, .60, .80, .90]))
print(df.corr())

# matplotlib绘制图表
fig, ax = plt.subplots(2, 2, figsize=(10, 5))

ax[0][0].hist(df['花瓣宽度'], color='black')
ax[0][0].set_ylabel('数量', fontsize=12, fontproperties=cnFont)
ax[0][0].set_xlabel('宽度', fontsize=12, fontproperties=cnFont)
ax[0][0].set_title('花瓣宽度分布', fontproperties=cnFont)

ax[0][1].hist(df['花瓣长度'], color='black')
ax[0][1].set_ylabel('数量', fontsize=12, fontproperties=cnFont)
ax[0][1].set_xlabel('长度', fontsize=12, fontproperties=cnFont)
ax[0][1].set_title('花瓣长度分布', fontproperties=cnFont)

ax[1][0].hist(df['花萼宽度'], color='black')
ax[1][0].set_ylabel('数量', fontsize=12, fontproperties=cnFont)
ax[1][0].set_xlabel('宽度', fontsize=12, fontproperties=cnFont)
ax[1][0].set_title('花萼宽度分布', fontproperties=cnFont)

ax[1][1].hist(df['花萼长度'], color='black')
ax[1][1].set_ylabel('数量', fontsize=12, fontproperties=cnFont)
ax[1][1].set_xlabel('长度', fontsize=12, fontproperties=cnFont)
ax[1][1].set_title('花萼长度分布', fontproperties=cnFont)

plt.show()