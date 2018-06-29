
# coding: utf-8

# # 库应用

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


# # 数据导入

# In[2]:


train_data = pd.DataFrame(data = load_digits().data, columns = ["columns"+str(x) for x in range(load_digits().data.shape[1])])
target_data = pd.DataFrame(data = load_digits().target, columns = ['target'])


# In[3]:


x_train,x_test,y_train,y_test = train_test_split(train_data,target_data,test_size=0.3, shuffle =True)


# In[4]:


# x_train = x_train.sort_index()
# x_test = x_test.sort_index()
# y_train = y_train.sort_index()
# y_test = y_test.sort_index()


# # 基本函数

# In[5]:


# 挑选出有处理价值的特征集合,要是某特征的所有样本的值都是一样的,那么就没有必要来求它的中点了;
def Select_columns(data):
    col = []
    for i in data.columns:
        if len(np.unique(data[i])) != 1:
            col.append(i)
    return col


# In[6]:


# #用于挑选出有处理价值的维度;
# def Select_columns(data):
#     col = []
#     for i in data.columns:
#         if len(np.where(data[i] != 0)[0]) != 0:
#             col.append(i)
#     return col


# In[7]:


# 计算出每个特征对应的所有样本的中点,n个样本对应n-1个中点
def Mid_point(data,columns):
    columns_sort_data = {}
    for i in columns:
        columns_sort_data[i] = data[i].sort_values(ascending = False).values 
        # 将第i维特征的值按倒序排列/其实这里就可以做排重处理了,好歹少算几个加法和除法;
    data_mid_point = {}
    for i in columns:
        x = []
        for j in range(data.shape[0]-1):
            temp = (columns_sort_data[i][j]+columns_sort_data[i][j+1])/2 
            # 将排序后的数值两两求中值;
            x.append(temp)
        x_tr = np.asarray(x)
        if len(np.unique(x_tr)) >= 1:
            data_mid_point[str(i)] = np.unique(x_tr) 
            # 因为中点向量中重复值很多,所以有必要做一下排重;通过字典来存储计算好的中值;
    return data_mid_point


# $$Ent(D) = - \sum_{k=1}^{|{\cal y}|} \cal{P}_k\log_2P_k$$

# $$Gain(D,a) = Ent(D) - \sum_{v = 1}^V \frac{|D^\upsilon|}{|D|} Ent(D^\upsilon)$$

# ## 对于连续值特征

# $$T_a = \lbrace \frac{a^\cal{i} + a^{i+1}}{2}| 1 \leq i \leq n-1 \rbrace$$

# $$Gain(D,a) = max_{t}$$

# ## 计算信息熵及信息增益

# In[8]:


# 信息熵计算,输入为一个概率值,这个函数之后就被代替了
def Count_entropy(p):
    if p == 0:
        return 0
    # 当概率为0时,手动将信息熵置0
    else:
        return p*np.log2(p)


# In[9]:


# 信息增益计算,输入为标签数据,返回信息增益
def Count_E_D(target):
    example_num = float(target.shape[0])
    target_num_dict = {}
    # 这里直接将函数写死了,应该使用sklearn数据自带的data.target_name
    for i in range(10):
        target_num_dict['target'+str(i)+'_num'] = target[target == i].count()[0]
    p = np.asarray(target_num_dict.values())/example_num
    E_D = 0
    for i in p:
        E_D = E_D + Count_entropy(i)
    return -E_D


# In[10]:


# 信息增益计算,输入为一个概率向量
def Count_E_D_P(p):
    E_D = 0
    for i in p:
        E_D = E_D + Count_entropy(i)
    return -E_D


# ### 特征最佳划分点选取

# In[11]:


# 根据计算的中点向量,计算出每个特征对应的信息增益,再选出使得信息增益最大的划分点(这个最佳划分点,每个特征有一个)
def Count_E_D_i(train, target, columns_use):
    mid = Mid_point(train, columns_use)
    E_D = Count_E_D(target)
    columns = mid.keys()
    G_D_columns = {}
    num_target = float(len(target))
    for i in columns: # 取出一维特征;
        split_data =  mid[i]  # 取出该维特征对应的中点向量;
        G_D_select = {}       # 拿来缓存该维特征所有中点得出的信息增益,以便之后挑选最佳;
        for j in range(len(split_data)):
            index_train_bigger = train[train[i] > split_data[j]].index.sort_values(ascending = True)
            index_train_smaller = train[train[i] <= split_data[j]].index.sort_values(ascending = True)
            bigger_num = float(len(index_train_bigger))
            smaller_num = float(len(index_train_smaller))
            out_put_bigger_num = np.unique(target.loc[index_train_bigger],return_counts=True)[1]
            out_put_smaller_num = np.unique(target.loc[index_train_smaller], return_counts = True)[1]
            p_bigger = out_put_bigger_num / bigger_num
            p_smaller = out_put_smaller_num / smaller_num
            E_D_bigger = Count_E_D_P(p_bigger)
            E_D_smaller = Count_E_D_P(p_smaller)
            G_D = E_D - (E_D_bigger*(bigger_num / num_target) + E_D_smaller*(smaller_num / num_target))
            G_D_select[str(j)] = G_D
        G_D_columns[i] = max(G_D_select.items(), key=lambda x: x[1]) 
        # 对该维特征的每一个中点都计算了信息增益之后,取出最大值;
    if len(G_D_columns) != 0: 
        # 得到所有维度的最佳划分点,这个if其实没啥用;
        return pd.DataFrame(G_D_columns), mid  
    # 习惯性将字典变为DataFrame,同时也返回一下中值字典,方便后便使用;


# In[12]:


# 根据最佳信息增益,以及中值字典,选出信息增益最大的特征,作为一个节点;
def Select_Node(data, mid_point):
    max_ed = data.loc[1].max()
    temp = np.where((data.loc[1] == data.loc[1].max()) == True)[0][0]
    max_col = data.columns[temp]
    slide_values_index = int(data[max_col][0])
    slide_values = mid_point[max_col][slide_values_index]
    return max_col, slide_values


# In[13]:


# 根据选取的节点,以及划分值,对数据集进行划分
def Slide_data(train, target, max_col, slide_values):
    index_bigger = train[train[max_col] >= slide_values].index.sort_values(ascending = True)
    index_smaller = train[train[max_col] < slide_values].index.sort_values(ascending = True)
    train_bigger = train.loc[index_bigger]
    train_smaller = train.loc[index_smaller]
    target_bigger = target.loc[index_bigger]
    target_smaller = target.loc[index_smaller]
    return train_bigger, target_bigger, train_smaller, target_smaller


# In[14]:


# 创建树节点的过程,和离散型特征不同的是,连续型特征,可以重复使用,阈值不同即可,所以递归的判定条件很关键;
def Creat_Tree(train, target):
    # 将递归的判定条件定为:如果依据之前节点划分出来的数据,都是同一个label,那么就往回返值,返回的值为这个label
    if np.unique(target.loc[train.index],return_counts=True)[1][0] == len(target.loc[train.index]):
        return target.loc[train.index].values[0][0]
    col = Select_columns(train) 
    # 每次要开始选取节点之前,先筛选一下特征;
    G_D, mid = Count_E_D_i(train, target, col) 
    # 得到每个维度的最佳信息增益以及中值字典;
    max_col, slide_values = Select_Node(G_D, mid)  
    # 根据最大信息增益原则得到节点,以及划分阈值;
    myTree = {str(max_col)+'_'+str(slide_values):{}} 
    # 创建一个字典,来存节点名称,包括节点对应的特征名以及阈值;
    # 根据节点对应的特征名以及划分阈值,重新划分数据集,递归第一步;
    train_bigger, target_bigger, train_smaller, target_smaller = Slide_data(train, target, max_col, slide_values)
    z = []
    z.append(((train_bigger, target_bigger),(train_smaller, target_smaller)))
    for i in ([0,1]): 
        # 这里通过0,1来标识和阈值的大小关系,0为大于等于,1为小于;
        myTree[str(max_col)+'_'+str(slide_values)][i] = Creat_Tree(z[0][i][0], z[0][i][1])
    return myTree  
# 得到最后的节点字典


# In[15]:


# 预测函数,完成的工作是,通过得到的节点字典,确定数据的标签;也是一个递归的过程;
def Predict(tree,train):
    # 递归的终止条件为 得到的值已不再是字典类型,而是确切的整型数据,也就是说,摸到了标签;
    if  type(tree).__name__ == 'int64':
        result = pd.DataFrame(index = train.index, data = float(tree), columns = ['target'])
        z.append(result) 
        # 定义全局变量,来存储预测结果;
    else:
        # 参考之前节点字典的形式,keys为:"columns name"_"slide values"
        key1 = tree.keys()[0]
        node1 = key1.split('_')[0] 
        # 取出特征名
        values1 = float(key1.split('_')[1]) 
        # 取出划分阈值
        index_bigger = train[train[node1] >= values1].index
        # 根据当前节点对应的特征及阈值,划分数据,准备递归
        index_smaller = train[train[node1] < values1].index
        Predict(tree[key1][0],train.loc[index_bigger]) 
        # 递归划分好的数据集
        Predict(tree[key1][1], train.loc[index_smaller]) 
        # 最终得到一个list,其中每个元素为:一个标签,及该标签对应的数据


# In[16]:


# 将这些分开的DataFrame放到一块,通过concat;
def to_pre(result):
    prediction = pd.DataFrame()
    for i in range(len(result)):
        if len(result[i].values) != 0:
            prediction = pd.concat([prediction, result[i]])
            
    return prediction.sort_index() 
# 根据index排序一下,好看一点zuojiedianle


# In[17]:


# 预测函数;注意到,我们的数据是经过划分,分为训练和测试部分的,sklearn中的train_test_split打乱了顺序,如果直接按
# sklearn 中的accuaracy_score(prediction.values(),y_test.vlaues()),那么由于顺序打乱了,所以就拉闸了;所以稍微操作一下
def Acc(x, x_pre):
    ind = x_pre.index
    return accuracy_score(x.loc[ind].values,x_pre.values)


# In[18]:


# 模型存储函数,每次都重新学很费时间
def save_model(T,filename):
    fw = open(filename,'w')
    pickle.dump(T,fw)
    fw.close


# In[19]:


# 模型读取
def load_model(filename):
    fr = open(filename)
    return pickle.load(fr)


# # 训练函数

# ## 训练过程:
# 1. 得到决策树的最终节点字典:
#     1. 判断输入的数据是否全为一种类型,是的话,返回标签
#     2. 计算每个维度的最佳信息增益以及中值字典,最佳信息增益中包含了特征名、最大信息增益的值以及所得信息增益依据的中点值在中值字典中的位置;
#         - 这个最大信息增益指的是同一个维度,采用不同中指点计算得到的信息增益中的最大值,所以最后我们得到的是一个诸多特征的最大信息增益表;
#     3. 通过上一步得到的特征名、以及中值位置信息,得到划分节点,以及对应的阈值;
#         - 这一部中,也有一个选取最大信息增益的过程,这个最大,对应的是不同特征之间的最大信息增益,直接就拿来做节点了;
#     4. 得到了划分节点以及划分阈值后,将数据进行划分,开始递归;
#     5. 得到最终的节点字典;
#         - 这个节点字典里,基本形式为"columns_name"_"slide_values",即包含了作为节点的特征名,也包含了划分阈值;
# 2. 存储模型参数,也就是学到的节点字典;
# 3. 根据学到的节点字典,来对测试集数据进行预测:
#     1. 同样也是一个递归过程,先判断一下,是不是已经到叶子了,到了的话,返回叶子的标签,以及到达那个叶子的数据的index,存到全局变量z中：
#         - 这里我把数据直接转换成了DataFrame形式
#     2. 把z中的零碎的DataFrame,整合一下,concat:
#         - 注意到,这个z是根据标签和index来创建的,所以,要是到那个叶子的数据为0,对应到z里边的元素就是一个空的DataFrame了,所以处理一下
# 4. 调用预测函数,得到准确率:
#     1. 需要注意一下,我的做法中,始终是使用数据在原始集合中的index来确认训练数据和标签之间的对应关系,所以,到最后计算准确率的时候也是一样,一定要记得index对应;

# In[20]:


# 训练函数
def Model_initial(x_train,y_train,x_test,y_test):
    tree = Creat_Tree(x_train, y_train)
    save_model(tree,'model2.txt')
    Predict(tree, x_test)
    prediction = to_pre(z)
    acc = Acc(prediction, y_test)
    return acc


# In[21]:


def Model_weight(x_test, y_test):
    tree = load_model('model2.txt')
    Predict(tree,x_test)
    prediction = to_pre(z)
    acc = Acc(prediction, y_test)
    return acc


# In[22]:


z = []


# In[23]:


Model_initial(x_train, y_train,x_test,y_test)

