   * [视频点击预测大赛](#视频点击预测大赛)
   * [基于点击率预估的推荐](#基于点击率预估的推荐)
      * [传统方法的局限](#传统方法的局限)
      * [点击率预估所采用的推荐方式](#点击率预估所采用的推荐方式)
   * [集成学习](#集成学习)
      * [boosting](#boosting)
         * [GBDT](#gbdt)
         * [xgboost](#xgboost)
         * [lightgbm](#lightgbm)
      * [bagging](#bagging)
      * [stacking](#stacking)
   * [视频点击预测大赛代码分析](#视频点击预测大赛代码分析)
      * [数据探索](#数据探索)
      * [特征工程](#特征工程)
         * [历史信息](#历史信息)
         * [时间差](#时间差)
         * [交叉特征](#交叉特征)
         * [embedding](#embedding)
      * [训练](#训练)

# 视频点击预测大赛

比赛地址：https://www.turingtopia.com/competitionnew/detail/e4880352b6ef4f9f8f28e8f98498dbc4/sketch 

数据下载地址也在比赛链接中，下载到当前目录下的dataset文件夹中

队伍名：起个什么名字好呢

最终排名：20名

得分：0.81239

# 基于点击率预估的推荐

## 传统方法的局限

推荐系统的传统方法很多如协同过滤等其实很难用于生产的系统
面对海量的用户数据时，协同过滤不能很好地解决数据稀疏性的问题和满足实时性的要求。
当一个系统如电商平台的用户数据过多时，大部分用户都是非活跃用户，非活跃用户点击的商品也很少，构建矩阵时矩阵会非常稀疏，难以确定与用户相似的其他用户，也难以确定与用户点击商品相似的其他商品。
另外，协同过滤每次都要实时计算所有用户和商品，也难以做到真正的“实时”推荐。

## 点击率预估所采用的推荐方式

例如视频的点击预测问题就对实时性要求较高，此时LR、GBDT、FM，以及基于深度学习神经网络的模型如deepfm、deepctr等模型显得更具优势。
在视频点击预测大赛中，主要采用基于LightGBM的模型。

# 集成学习

有监督学习中想训练出一个效果特别好的单一模型比较难，但是可以通过将多个弱监督模型组合在一起，构成一个强监督模型。
集成学习主要包括三种方法：

- boosting，梯度提升
- bagging，有放回抽样
- stacking，融合

## boosting

boosting方法包括两方面：

- 如何改变数据的权值使得效果不好的数据得到重视
- 如何将多个弱分类器组合成一个强分类器

boosting的工作算法步骤：

1. 首先赋予每个训练样本相同的权值，在训练数据中用初始权值训练一个弱学习器，然后根据弱学习器的误差来更新训练样本的权重，使得之前学习误差高的数据权值增大，进而这些误差高的点在后续得到更多重视。
2. 基于调整后的训练集训练弱学习器2
3. 将弱学习器进行组合得到最终的强学习器

可见boosting是串行的，后面的学习器基于前面学习器的权值。

常用的有GBDT、LightGBM、Xgboost等。

### GBDT

GBDT算法可以看成是由K棵树组成的加法模型
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAyMC9wbmcvNzA1NDYxLzE1ODEyMzM3OTYzMjctZmFiZGE2NzAtOGM3Zi00MmUwLTk1NjktYmNmMGU1YzJjMzQzLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=65&name=image.png&originHeight=88&originWidth=177&size=3776&status=done&style=none&width=131)
GBDT与传统的boosting有些不同，GBDT的每一次计算都是为了减少上一次的残差。而为了消除残差，我们可以在残差减小的梯度方向上建立模型，所以说，在GradientBoost中，每个新的模型的建立是为了使得之前的模型的残差往梯度下降的方法。
![image.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAyMC9wbmcvNzA1NDYxLzE1ODEyMzI4NDAxMjQtNTNlMzk5OWUtMzUzZC00NmUzLTlkOTMtOTU0YzM5NTE0YWE3LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=355&name=image.png&originHeight=548&originWidth=818&size=63511&status=done&style=none&width=530)

### xgboost

xgboost vs GBDT
xgboost可以并行，Boosting不是一种串行的结构吗?怎么并行 
的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代

中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。

### lightgbm

1. **xgboost采用的是level-wise的分裂策略，而lightGBM采用了leaf-wise的策略**，区别是xgboost对每一层所有节点做无差别分裂，可能有些节点的增益非常小，对结果影响不大，但是xgboost也进行了分裂，带来了无必要的开销。

leaft-wise的做法是在当前所有叶子节点中选择分裂收益最大的节点进行分裂，如此递归进行，很明显leaf-wise这种做法容易过拟合，因为容易陷入比较高的深度中，因此需要对最大深度做限制，从而避免过拟合。

1. lightgbm使用了基于histogram的决策树算法，这一点不同与xgboost中的 exact 算法，histogram算法在内存和计算代价上都有不小优势。

- 内存上优势：很明显，直方图算法的内存消耗为(#data* #features * 
  1Bytes)(因为对特征分桶后只需保存特征离散化之后的值)，而xgboost的exact算法内存消耗为：(2 * #data * 
  #features* 
  4Bytes)，因为xgboost既要保存原始feature的值，也要保存这个值的顺序索引，这些值需要32位的浮点数来保存。
- 计算上的优势，预排序算法在选择好分裂特征计算分裂收益时需要遍历所有样本的特征值，时间为(#data),而直方图算法只需要遍历桶就行了，时间为(#bin)

3. 直方图做差加速

一个子节点的直方图可以通过父节点的直方图减去兄弟节点的直方图得到，从而加速计算。

4. lightgbm支持直接输入categorical 的feature

在对离散特征分裂时，每个取值都当作一个桶，分裂时的增益算的是**是否属于某个category**的gain。类似于one-hot编码。

5. 多线程优化

## bagging

bagging是Bootstrap Aggregating，称为自助法，是一种有放回的抽样方法。
每个弱学习器从训练样本中随机采样然后训练，最后采取投票的方式解决分类问题、采取平均的方式解决回归问题。
bagging中每个学习器之间没有依赖关系。

## stacking

stacking是指训练一个模型用于组合其他的模型。
其中，基础训练模型是基于完整训练集进行训练的，而元模型是基于基础训练模型的结果进行训练。

# 视频点击预测大赛代码分析

该比赛提供了历史上前三天某用户的点击数据和个人的终端等信息，需要我们判断未来一天内会不会发生点击。
实际上需要预测的就是点击(1)还是不点击(0)的一个分类问题。
主要利用四组特征：

- 历史信息，即前一天的点击量、曝光量、点击率；
- 前x次曝光、后x次曝光到当前的时间差，后x次到当前曝光的时间差是穿越特征，并且是最强的特征；
- 二阶交叉特征；
- embedding。

## 数据探索

[https://github.com/sherpahu/VideoClickCompetition20th/blob/master/DataExplore.py](https://github.com/sherpahu/VideoClickCompetition20th/blob/master/DataExplore.py)
来自[https://www.turingtopia.com/models/details/notebook/92514425768d4b3097b170ff96df0dfc](https://www.turingtopia.com/models/details/notebook/92514425768d4b3097b170ff96df0dfc)

## 特征工程

### 历史信息

```python
for f in [
    ['deviceid'],
    ['pos', 'deviceid'],
    # ...
]:
    print('------------------ {} ------------------'.format('_'.join(f)))
    
    # 对前一天的点击次数进行统计
    tmp = click_df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg({'_'.join(f) + '_prev_day_click_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_click_count'] = df['_'.join(f) + '_prev_day_click_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_click_count'] = None
    
    # 对前一天的曝光量进行统计
    tmp = df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg({'_'.join(f) + '_prev_day_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_count'] = df['_'.join(f) + '_prev_day_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_count'] = None
    
    # 计算前一天的点击率
    df['_'.join(f) + '_prev_day_ctr'] = df['_'.join(f) + '_prev_day_click_count'] / (
            df['_'.join(f) + '_prev_day_count'] + df['_'.join(f) + '_prev_day_count'].mean())

    del tmp
    print('runtime:', time.time() - t)
del click_df
df = reduce_mem(df)
```

### 时间差

曝光时间差越大代表这个视频越“成熟”，越有受人喜欢，越可能被点。
类似的可以参考[https://zhuanlan.zhihu.com/p/95418813](https://zhuanlan.zhihu.com/p/95418813)[第十名的分享](https://zhuanlan.zhihu.com/p/95418813)

```python
for f in [
    ['deviceid'], ['newsid'], ['lng_lat'],
    ['pos', 'deviceid'], ['pos', 'newsid'], ['pos', 'lng_lat'],     ['pos', 'device_vendor'],
    ['pos', 'deviceid', 'lng_lat'],     ['pos', 'device_vendor', 'lng_lat'],
    ['netmodel', 'deviceid'],
    ['pos', 'netmodel', 'deviceid'],
    ['netmodel', 'lng_lat'], ['deviceid', 'lng_lat'],
    ['netmodel', 'deviceid', 'lng_lat'], ['pos', 'netmodel', 'lng_lat'],
    ['pos', 'netmodel', 'deviceid', 'lng_lat']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    tmp = sort_df[f + ['ts']].groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 5, 10]:
        sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            f + ['ts', '{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap), '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        ].drop_duplicates(f + ['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f + ['ts'], how='left')
        del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del tmp2

    del tmp
    df = reduce_mem(df)
    print('runtime:', time.time() - t)
del df['ts']
gc.collect()
```

### 交叉特征

常用的特征工程方法

```python
cross_cols = ['deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat']
for f in cross_cols:
    for col in cross_cols:
        if col == f:
            continue
        print('------------------ {} {} ------------------'.format(f, col))
        df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
            'cross_{}_{}_nunique'.format(f, col): 'nunique',
            'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0]) # 熵
        }), on=f, how='left')
        if 'cross_{}_{}_count'.format(f, col) not in df.columns.values and 'cross_{}_{}_count'.format(col, f) not in df.columns.values:
            df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                'cross_{}_{}_count'.format(f, col): 'count' # 共现次数
            }), on=[f, col], how='left')
        if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[f + '_count'] # 比例偏好
        if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / df[col + '_count'] # 比例偏好
        df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / df[f + '_count']
        print('runtime:', time.time() - t)
    df = reduce_mem(df)
del df['id']
gc.collect()
```

### embedding

利用gensim的Word2Vec进行embedding。
分组，得到某个device_id关联的视频id、地理位置的embedding

由某个device_id对应的视频id、地理位置可以提取出这个device_id与其余的关联关系，可以说代表了device_id对应的人的喜好等信息。

```python
def emb(df, f1, f2):
    emb_size = 8
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, sg=0, hs=1, seed=2019)
    emb_matrix = []#np.array([])
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix=np.array(emb_matrix)
    for i in range(emb_size):
        print('{}_{}_emb_{}'.format(f1, f2, i))
        print(emb_matrix[:, i])
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = reduce_mem(tmp)
    print('runtime:', time.time() - t)
    return tmp


emb_cols = [
    ['deviceid', 'newsid'],
    ['deviceid', 'lng_lat'],
    ['newsid', 'lng_lat'],
    ['pos', 'deviceid'],
    # ...
]
for f1, f2 in emb_cols:
    df = df.merge(emb(sort_df, f1, f2), on=f1, how='left')
    df = df.merge(emb(sort_df, f2, f1), on=f2, how='left')
del sort_df
gc.collect()
```

## 训练

```python
print('=============================================== training validate ===============================================')
fea_imp_list = []
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=5000,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019,
    metric=None
)

print('************** training **************')
clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** validate predict **************')
best_rounds = clf.best_iteration_
best_auc = clf.best_score_['valid_0']['auc']
val_pred = clf.predict_proba(val_x)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print('=============================================== training predict ===============================================')
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=best_rounds,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019
)

print('************** training **************')
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    categorical_feature=cate_cols,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** test predict **************')
sub = pd.read_csv('./dataset/sample.csv')
sub['target'] = clf.predict_proba(test_df)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)
```