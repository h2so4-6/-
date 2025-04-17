# user_id:用户ID，order_dt:购买日期，order_products:购买产品数量,order_amount:购买金额
# 数据时间：1997年1月~1998年6月用户行为数据，约6万条,一共69659条
import numpy as np          # 计算功能
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')  # 更改绘图风格，R语言绘图库的风格
plt.rcParams['font.sans-serif'] = ['SimHei']



# 导入数据
columns = ['user_id','order_dt','order_products','order_amount']            # columns是列的意思，这里我们来给txt设置列名
df = pd.read_table("E:/python项目实战/3、用户消费行为数据分析/1-项目介绍,需求分析/资料/CDNOW_master.txt",names=columns,sep=r'\s+')  #sep:'\s+':匹配任意个空格

print(df.head())            # 显示出前几行的数据用来观察，因为数据太多
# 1.日期格式需要转换
# 2.存在同一个用户一天内购买多次行为

print(df.describe())                #查看详情，显示出来有总数，最大值，最小值，平均值，中间值等等
# 1.用户平均每笔订单购买2.4个商品，标准差2.3，稍微有点波动，属于正常。
#然而75%分位数的时候，说明绝大多数订单的购买量都不多，围绕在2~3个产品左右；
# 2.购买金额，反映出大部分订单消费金额集中在中小额，30~45左右

print(df.info())        # df.info() 是 Pandas 中一个非常实用的方法，用于快速查看 DataFrame 的简要摘要信息，包括数据类型、非空值数量、内存占用等.运行 df.info() 会输出以下关键信息：
# 数据行数和列数（RangeIndex 和列名列表）。
# 每列的非空值数量（Non-Null Count），可快速发现缺失值。
# 每列的数据类型（Dtype），如 int64、float64、object（通常是字符串）等。
# 内存占用（memory usage），帮助优化大数据集的性能。
# 从中可以得知没有空值




# 数据预处理
df['order_date'] = pd.to_datetime(df['order_dt'],format='%Y%m%d')
# format参数：按照指定的格式去匹配要转换的数据列。
# %Y:四位的年份1994   %m:两位月份05  %d:两位月份31
# %y：两位年份94  %h:两位小时09  %M：两位分钟15    %s:两位秒
# 将order_date转化成精度为月份的数据列
df['month'] = df['order_date'].values.astype('datetime64[M]')  #[M] :控制转换后的精度
print(df.head())            # 月份那一栏被设置成年——月——日格式，且默认为每月一号
print(df.info())


# 用户整体消费趋势分析（按月份）
matplotlib.use('TkAgg')
# 按月份统计产品购买数量，消费金额，消费次数，消费人数
plt.figure(figsize=(20, 15))  # ，设置图片的长宽，单位是英寸
# 子图1：每月的产品购买数量
plt.subplot(2, 2, 1)  # 两行两列，第一个位置
df.groupby('month')['order_products'].sum().plot(kind='line')
plt.title('每月的产品购买数量')
plt.xlabel('月份')
plt.ylabel('购买数量')
# 子图2：每月的消费金额
plt.subplot(2, 2, 2)  # 两行两列，第二个位置
df.groupby('month')['order_amount'].sum().plot(kind='line')
plt.title('每月的消费金额')
plt.xlabel('月份')
plt.ylabel('金额')
# 子图3：每月的消费次数
plt.subplot(2, 2, 3)  # 两行两列，第三个位置
df.groupby('month')['user_id'].count().plot(kind='line')
plt.title('每月的消费次数')
plt.xlabel('月份')
plt.ylabel('次数')
# 子图4：每月的消费人数
plt.subplot(2, 2, 4)  # 两行两列，第四个位置
df.groupby('month')['user_id'].nunique().plot(kind='line')  # 使用nunique()替代apply更高效，nunique() 是 Pandas 中的一个常用方法，用于 计算唯一值（去重）的数量。它的名称是 "number of unique" 的缩写
plt.title('每月的消费人数')
plt.xlabel('月份')
plt.ylabel('人数')
# 调整子图间距
plt.tight_layout()
# 显示图形
plt.show()
#分析结果：
# 图一可以看出，前三个月销量非常高，而以后销量较为稳定，并且稍微呈现下降趋势
# 图二可以看出,依然前三个月消费金额较高，与消费数量成正比例关系，三月份过后下降严重，并呈现下降趋势，思考原因？1：跟月份有关，
# 在我国来1，2，3月份处于春节前后。2.公司在1，2，3，月份的时候是否加大了促销力度
# 图三可以看出，前三个月订单数在10000左右，后续月份的平均消费单数在2500左右
# 图四可以看出，前三个月消费人数在8000~10000左右，后续平均消费消费在2000不到的样子
# 总结：所有数据显示，97年前三月消费事态异常，后续趋于常态化



# 用户消费金额，消费次数(产品数量)描述统计
user_grouped = df.groupby(by='user_id').agg({'order_dt': 'sum','order_products': 'sum', 'order_amount': 'sum'})     # 这里一定要用agg指出显示哪些列，不然默认显示所有列，而order_date和month的类型是不能相加的datetime64类型，所以会报错
print(user_grouped.describe())
print(f"总用户数量: {len(user_grouped)}")            # 上一模块已经去重
# 从用户的角度：用户数量23570个，每个用户平均购买7个CD，但是中位数只有3，
# 并且最大购买量为1033，平均值大于中位数，属于典型的右偏分布(替购买量<7的用户背锅)
# 从消费金额角度：平均用户消费106，中位数43，并且存在土豪用户13990，结合分位数和最大值来看，平均数与75%分位数几乎相等，
# 属于典型的右偏分布，说明存在小部分用户（后面的25%）高额消费（这些用户需要给消费金额<106的用户背锅，只有这样才能使平均数维持在106）

#绘制每个用户的产品的购买量与消费金额散点图
df.plot(kind='scatter',x='order_products',y='order_amount')     # kind='scatter'意思是绘制图类型为散点图
plt.show()
# 从图中可知，用户的消费金额与购买量呈现线性趋势，每个商品均价15左右
# 订单的极值点比较少（消费金额>1000，或者购买量大于60）,对于样本来说影响不大，可以忽略不记。


# 用户消费分布图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.xlabel('每个订单的消费金额')
# kind='hist'意思是绘制图类型为直方图
df['order_amount'].plot(kind='hist',bins=50)  # bins:区间分数，影响柱子的宽度，值越大柱子越细，精度更准。宽度=（列最大值-最小值）/bins
# 消费金额在100以内的订单占据了绝大多数
plt.subplot(1,2,2)
plt.xlabel('每个user_id购买的数量')
df.groupby(by='user_id')['order_products'].sum().plot(kind='hist',bins=50)
# 调整子图间距
plt.tight_layout()
# 显示图形
plt.show()
# 图二可知，每个用户购买数量非常小，集中在50以内
# 两幅图得知，我们的用户主要是消费金额低，并且购买小于50的用户人数占据大多数（在电商领域是非常正常的现象）



# 用户累计消费金额占比分析（用户的贡献度）
#进行用户分组，取出消费金额，进行求和，排序，重置索引
user_cumsum = df.groupby(by='user_id')['order_amount'].sum().sort_values().reset_index()
print(user_cumsum)



#用户分组，取最小值，即为首购时间，
# A  1997-1-1
# B  1997-1-1
# 1997-1-1   ?（2个）
df.groupby(by='user_id')['order_date'].min().value_counts().sort_index().plot()     # value_counts()是对值进行数量统计
plt.title('首次购买')
plt.xlabel('首次购买的时间')
plt.ylabel('用户量')
plt.show()
# 由图可知，首次购买的用户量在1月1号~2月10号呈明显上升趋势，后续开始逐步下降，猜测：有可能是公司产品的推广力度或者价格调整所致
df.groupby(by='user_id')['order_date'].max().value_counts().sort_index().plot()
plt.title('最后一次购买')
plt.xlabel('最后一次购买的时间')
plt.ylabel('用户量')
plt.show()
# 大多数用户最后一次购买时间集中在前3个月，说明缺少忠诚用户。
# 随着时间的推移，最后一次购买商品的用户量呈现上升趋势，猜测：这份数据选择是的前三个月消费的用户在后面18个月的跟踪记录



# 用户分层
# 构建RFM模型
# Recency（最近一次消费）：指的是客户最近一次购买的时间距离现在的时间间隔。最近购买的客户通常更有可能再次购买，因为他们对品牌的记忆和认知还比较新鲜。
# 通过对客户最近购买行为的分析，可以识别出哪些客户是近期活跃的，哪些客户可能已经流失，R越小，交易日期越近。
# Frequency（消费频率）：指客户在特定时间段内购买的次数。购买频率高的客户通常对品牌有较高的忠诚度，他们可能会成为品牌的长期客户。
# 通过对客户购买频率的分析，可以了解客户的购买习惯和忠诚度，从而制定相应的营销策略。F越大，交易越频繁。
# Monetary（消费金额）：指客户在特定时间段内的总购买金额。消费金额高的客户通常是品牌的高价值客户，他们对品牌的贡献较大。
# 通过对客户购买金额的分析，可以识别出哪些客户是高价值客户，从而给予他们更多的关注和优待。M越大，客户价值越高。

# 透视表：对数据动态排布并且分类汇总的表格格式
# 透视表的使用（index:相当于groupby,values:取出的数据列，aggfunc:key值必须存在于values列中，并且必须跟随有效的聚合函数）
rfm = df.pivot_table(index='user_id',
                    values=['order_products','order_amount','order_date'],
                    aggfunc={                               # aggfunc聚合函数
                        'order_date':'max',# 最后一次购买
                        'order_products':'sum',# 购买产品的总数量
                        'order_amount':'sum'  # 消费总金额
                        })
print(rfm.head())

# 用每个用户的最后一次购买时间-日期列中的最大值，最后再转换成天数，小数保留一位
rfm['R'] = -(rfm['order_date']-rfm['order_date'].max())/np.timedelta64(1,'D')  # 取相差的天数，保留一位小数，精确到1位，D表示‘天’
rfm.rename(columns={'order_products':'F','order_amount':'M'},inplace=True)      # 更换列名    ,inplace表示更换，不能少
print(rfm.head())


# RFM计算方式：每一列数据减去数据所在列的平均值，有正有负，根据结果值与1做比较，如果>=1,设置为1，否则0
def rfm_func(x):  # x:分别代表每一列数据
    level = x.apply(lambda x: '1' if x >= 1 else '0')
    label = level['R'] + level['F'] + level['M']  # 举例：100    001
    d = {
        '111': '重要价值客户',
        '011': '重要保持客户',
        '101': '重要发展客户',
        '001': '重要挽留客户',
        '110': '一般价值客户',
        '010': '一般保持客户',
        '100': '一般发展客户',
        '000': '一般挽留客户'

    }
    result = d[label]
    return result
# rfm['R']-rfm['R'].mean()   这一步是计算出数值与数值平均值的差值，然后调用定义的函数rfm_func来判断是哪种客户
rfm['label'] = rfm[['R', 'F', 'M']].apply(lambda x: x - x.mean()).apply(rfm_func, axis=1)
# axis=0：沿着行的方向（纵向）操作，即对每一列进行处理。
# axis=1：沿着列的方向（横向）操作，即对每一行进行处理。
print(rfm.head())

# 客户分层可视化
for label,grouped in rfm.groupby(by='label'):
#print(label,grouped)
    x = grouped['F']  # 单个用户的购买数量
    y = grouped['R']  # 最近一次购买时间与98年7月的相差天数
    plt.scatter(x,y,label=label)        # scatter散点图
plt.legend()  #显示图例
plt.xlabel('F')
plt.ylabel('R')
plt.show()




# 新老，活跃，回流用户分析
# - 新用户的定义是第一次消费。
# - 活跃用户即老客，在某一个时间窗口内有过消费。
# - 不活跃用户则是时间窗口内没有消费过的老客。
# - 回流用户：相当于回头客的意思。
# - 用户回流的动作可以分为自主回流与人工回流，自主回流指玩家自己回流了，而人工回流则是人为参与导致的。
pivoted_counts = df.pivot_table(         # 透视表
                index='user_id',
                columns ='month',
                values = 'order_dt',
                aggfunc ='count'      # aggfunc聚合函数  将用户的消费日期求和,!!!aggfunc函数只能对values中指定的列进行聚合
).fillna(0)         # 把（NA）空值填充成0
print(pivoted_counts.head())

# 由于浮点数不直观，并且需要转成是否消费过即可，1表示消费过，0表示没消费
df_purchase = pivoted_counts.map(lambda x:1 if x>0 else 0)
print(df_purchase.head())
# apply:作用于dataframe数据中的一行或者一列数据。
# apply() 与map（）相比更灵活，可以用于 Series 或 DataFrame：
# 在 Series 上，apply() 和 map() 类似，但 apply() 可以接受更复杂的函数。
# 在 DataFrame 上，apply() 可以按行或列操作。
# DataFrame 是一种二维表格型数据结构

# applymap:作用于dataframe数据中的每一个元素
# DataFrame.applymap() 方法在未来的 Pandas 版本中将被弃用，官方推荐改用 DataFrame.map()

# map:本身是一个series的函数，在df结构中无法使用map函数，map函数作用于series中每一个元素的。
# Series 本身是一种一维数组结构，可以存储不同类型的数据（数值、字符串、布尔值等）。
# Series 是 Pandas 的核心数据结构之一，类似于带标签的一维数组。它的数据类型（dtype）可以是：
# 数值类型：int, float, bool
# 字符串类型：object（Pandas 中用 object 表示字符串）
# 时间类型：datetime64, timedelta64
# 分类类型：category
# 其他：如 complex, bytes 等（较少用）


def active_status(data):  # data：每一行数据（共18列）
    status = []  # 一个空列表 存储用户18个月的状态（new|active|unactive|return|unreg） 新用户，活跃用户，不活跃用户，回流用户，未注册用户
    # 新用户：第一次消费
    # 活跃用户：上个月消费，这个月也消费
    # 不活跃用户：之前消费过，这个月没消费
    # 回流用户:之前消费过，上月没消费，这个月又消费了
    # 未注册用户：一直没消费过

# 你正在使用类似 data[i] 的方式通过整数位置（position）访问 Series 的值。
# Pandas 未来版本将不再支持直接用整数索引（data[i]）按位置访问，而是会统一按标签（label）处理（和 DataFrame 行为一致）。
    for i in range(18):
        # 判断本月没有消费==0
        if data.iloc[i] == 0:
            if len(status) == 0:  # 前几个月没有任何记录（也就是97年1月==0）
                status.append('unreg')
            else:  # 之前的月份有记录（判断上一个月状态）
                if status[i - 1] == 'unreg':  # 一直没有消费过
                    status.append('unreg')
                else:  # 上个月的状态可能是：new|active|unative|reuturn
                    status.append('unactive')
        else:  # 本月有消费==1
            if len(status) == 0:
                status.append('new')  # 第一次消费
            else:  # 之前的月份有记录（判断上一个月状态）
                if status[i - 1] == 'unactive':
                    status.append('return')  # 前几个月不活跃，现在又回来消费了，回流用户
                elif status[i - 1] == 'unreg':
                    status.append('new')  # 第一次消费
                else:  # new|active
                    status.append('active')  # 活跃用户

    return pd.Series(status, df_purchase.columns)  # 值：status,列名：18个月份

# axis=0：沿着行的方向（纵向）操作，即对每一列进行处理。
# axis=1：沿着列的方向（横向）操作，即对每一行进行处理。
purchase_states = df_purchase.apply(active_status, axis=1)  # 得到用户分层结果
print(purchase_states.head())

# 把unreg状态用nan替换
# pd.value_counts() 是 Pandas 中的一个函数，用于统计一个 Series（或数组）中每个唯一值出现的次数，并按计数从高到低排序
purchase_states_ct = purchase_states.replace('unreg',np.nan).apply(lambda x:pd.value_counts(x)) # apply列操作，对每一列的数据value_counts进行分类统计，这里的axis默认是0，不写
print(purchase_states_ct.head())

# 数据可视化，面积图
purchase_states_ct.fillna(0).T.plot.area(figsize=(12,6))  # fillna(0)的操作是把NA替换成0方便后续画图，填充nan之后，进行行列变换.T
plt.xlabel('时间')
plt.ylabel('人数')
plt.show()
# 由图可知：灰色区域是不活跃用户，占比较大
# 前三个月新用户和活跃用户占比较大
# 4月份过后，新用户和活跃用户，都呈现下降趋势，并且趋于平稳状态
# 回流用户主要产生在4月之后，呈稳定趋势，是网站的重要客户

#每月中回流用户占比情况（占所有用户的比例）
plt.figure(figsize=(12,6))
rate = purchase_states_ct.fillna(0).T.apply(lambda x:x/x.sum(),axis=1)
# axis=0：沿着行的方向（纵向）操作，即对每一列进行处理。
# axis=1：沿着列的方向（横向）操作，即对每一行进行处理。
plt.plot(rate['return'],label='return')     # label是标签，rate['return']是取出回流用户的计算出来的当月在18个月中的占比值
plt.plot(rate['active'],label='active')
plt.legend()        # 加图例，比如：图表右上角会显示一个图例框，标明 Line A 和 Line B 对应的线条样式和颜色
plt.show()
# 由图可知，前3个月，活跃用户占比比较大，维持在7%左右，而回流用户比例在上升，由于new用户还没有足够时间变成回流用户
# 4月份过后，不论是活跃用户，还是回流用户都呈现出下降趋势，但是回流用户依然高于活跃用户。


# 用户的购买周期
# shift函数：将数据移动到一定的位置
# axis=0：沿着行的方向（纵向）操作，即对每一列进行处理。
# axis=1：沿着列的方向（横向）操作，即对每一行进行处理。

# 计算购买周期（购买日期的时间差值）
order_diff = df.groupby(by='user_id')['order_date'].apply(lambda x:x-x.shift()) #当前订单日期-上一次订单日期,.shift(axis=0)整体向下移动一个位置（默认值：axis=0）
print(order_diff.describe())

(order_diff/np.timedelta64(1,'D')).hist(bins = 20) #影响柱子的宽度，  每个柱子的宽度=（最大值-最小值）/bins
# order_diff/np.timedelta64(1,'D')是转换成精度为1的天数
plt.xlabel('购买周期')
plt.ylabel('用户数')
plt.title('用户的购买周期直方图')
plt.show()
# 得知：平均消费周期为68天
# 大多数用户消费周期低于100天
# 呈现典型的长尾分布，只有小部分用户消费周期在200天以上（不积极消费的用户），可以在这批用户消费后3天左右进行电话回访后者短信
# 赠送优惠券等活动，增大消费频率




# 用户的购物生命周期
#计算方式：用户最后一次购买日期(max)-第一次购买的日期(min)。如果差值==0，说明用户仅仅购买了一次
user_life = df.groupby('user_id')['order_date'].agg(['min','max'])      # agg取出最小值和最大值，可以同时操作很多值
(user_life['max']==user_life['min']).value_counts().plot.pie(autopct='%1.1f%%') #格式化成1为小数
# 若两值相等，则仅消费一次，反之则多次消费（这里是布尔用法），True：12054   False：11516
plt.legend(['仅消费一次','多次消费'])
plt.title('用户消费情况饼图')
plt.show()
#一半以上的用户仅仅消费了一次，说明运营不利，留存率不好

print((user_life['max']-user_life['min']).describe()) #生命周期分析
#用户平均生命周期为134天，但是中位数==0，再次验证大多数用户消费了一次，低质量用户。
# 75%分位数以后的用户，生命周期>294天，属于核心用户，需要着重维持。
# 前三个月的新用户数据，所以分析的是这些用户的生命周期


### 绘制所有用户生命周期直方图+多次消费
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
((user_life['max']-user_life['min'])/np.timedelta64(1,'D')).hist(bins=15)
plt.title('所有用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')

plt.subplot(1,2,2)
u_1 = (user_life['max']-user_life['min']).reset_index()[0]/np.timedelta64(1,'D')        # reset_index()加上了一列索引
u_1[u_1>0].hist(bins=15)        # 过滤出生命周期大于0的多次消费者
plt.title('多次消费的用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')
plt.show()
# 对比可知，第二幅图过滤掉了生命周期==0的用户，呈现双峰结构
# 虽然二图中还有一部分用户的生命周期趋于0天，但是比第一幅图好了很多，虽然进行了多次消费，但是不成长期
# 来消费，属于普通用户，可针对性进行营销推广活动
# 少部分用户生命周期集中在300~500天，属于我们的忠诚客户，需要大力度维护此类客户




## 复购率和回购率分析
# 复购率分析
#计算方式：在自然月内，购买多次的用户在总消费人数中的占比（若客户在同一天消费了多次，也称之复购用户）
#消费者有三种：消费记录>=2次的；消费中人数；本月无消费用户；
#复购用户:1    非复购的消费用户：0   自然月没有消费记录的用户：NAN(不参与count计数)
purchase_r = pivoted_counts.applymap(lambda x: 1 if x>1 else np.nan  if x==0 else 0) # 若复购则设置成1，若为零则设置为NA，其他情况设置成0，也就是非复购的（本月只买一次）消费用户
#purchase_r.sum() :求出复购用户
#purchase_r.count():求出所有参与购物的用户（NAN不参与计数）
(purchase_r.sum()/purchase_r.count()).plot(figsize=(12,6))
plt.title('每月的复购率')
plt.show()
# 前三个月复购率开始上升，后续趋于平稳维持在20%~22%之间。
# 分析前三个月复购率低的原因，可能是因为大批新用户仅仅购买一次造成的。

# 回购率分析:
# 计算方式：在一个时间窗口内进行了消费，在下一个窗口内又进行了消费
def purchase_back(data):
    status = [] #存储用户回购率状态
    #1:回购用户   0：非回购用户（当前月消费了，下个未消费）   NaN:当前月份未消费
    for i in range(17):
        #当前月份消费了
        if data[i] == 1:
            if data[i+1]==1:
                status.append(1) #回购用户
            elif data[i+1] == 0: #下个月未消费
                status.append(0)
        else: #当前月份未进行消费
            status.append(np.nan)
    status.append(np.nan) #填充最后一列数据,因为对于最后一列数据来说，我们并不知道他的下一月是什么情况，所以设置成NA
    return pd.Series(status,df_purchase.columns)

purchase_b = df_purchase.apply(purchase_back,axis=1)
print(purchase_b.head())

#回购率可视化
plt.figure(figsize=(20,4))
plt.subplot(2,1,1)
#回购率
(purchase_b.sum() / purchase_b.count()).plot(label='回购率')
#复购率
(purchase_r.sum()/purchase_r.count()).plot(label='复购率')
plt.legend()
plt.ylabel('百分比%')
plt.title('用户回购率和复购率对比图')
#回购率可知，平稳后在30%左右，波形性稍微较大
#复购率低于回购率，平稳后在20%左右，波动小较小
#前三个月不困是回购还是复购，都呈现上升趋势，说明新用户需要一定时间来变成复购或者回购用户
#结合新老用户分析，新客户忠诚度远低于老客户忠诚度。

#回购人数与购物总人数
plt.subplot(2,1,2)
plt.plot(purchase_b.sum(),label='回购人数')
plt.plot(purchase_b.count(),label='购物总人数')
plt.xlabel('month')
plt.ylabel('人数')
plt.legend()
plt.show()
# 前三个月购物总人数远远大于回购人数，主要是因为很多新用户在1月份进了首次购买
# 三个月后，回购人数和购物总数开始稳定，回购人数稳定在1000左右，购物总人数在2000左右。

