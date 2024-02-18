Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning

简介： 在实验和观察研究中，Uplift治疗效果越来越受到关注。我们描述了一些元算法，它们可以利用机器学习和统计学中的任何监督学习或回归方法来估计条件平均处理效果(CATE)函数。元算法建立在基本算法(如随机森林(RF)、贝叶斯加性回归树(BART)或神经网络)的基础上，估计CATE，这是一个基本算法不能直接估计的函数。我们引入了一种新的元算法，X-Learner，当一个治疗组的单元数量比另一个治疗组大得多时，它被证明是有效的，并且可以利用CATE函数的结构特性。例如，如果CATE函数是线性的，治疗和控制中的响应函数是Lipschitz连续的，则X-Learner在正则性条件下仍然可以实现参数率。然后我们介绍使用RF和BART作为基础学习器的X-Learner版本。

## 问题背景和定义

随着包含关于人类及其行为的细粒度信息的大型数据集的兴起， 企业和政策制定者对此越来越感兴趣干预效果如何因个体和环境而异。他们希望寻求估计条件平均治疗效果(CATE)来个性化政策制度和更好地理解因果机制。我们 引入一种新的估计器，叫做X-Learner，并描述它和许多其他CATE估计器 统一的元学习者框架。模型的效果比较 是基于广泛的模拟、理论和政治科学随机实地实验的两个数据上效果比较。 在第一个随机实验中，我们估计了邮件对选民投票率的影响，在第二个随机实验中， 我们衡量挨家挨户谈话的影响 对性别不一致个体的偏见。 在两个实验中，发现处理效果是不是固定数值，我们通过估计CATE来量化这种异质性。 为了估计CATE，我们建立在统计学和机器学习中的回归或监督学习方法上， 这些都成功地应用于广泛的应用中。具体来说，我们研究了在二元治疗设置中估计CATE的元算法(或元生成器)。元算法分解估计 CATE可以分解成几个子回归问题 用任何回归或监督学习方法求解



### 问题的框架和形式化定义





采用Neyman-Rubin 的潜在结果的模型框架(POF)，并假设一个超总体上的分布 P上产出N 个独立的样本。 X 表示协变量，Y表示两种干预下的潜在结果，W表示干预的indicator。
$$
\displaylines{
(Y_i(0),Y_i(1),X_i,W_i) \sim P \\
ATE := E[Y(1) - Y(0)] \\
\mu_0(X) := E[Y(0)|X=x]\text{    and   } \mu_1(X) := E[Y(0)|X=x]
}
$$



$$
\displaylines{
D_i = Y_i(1) - Y_0(0) \\
\tau(x) := E[Y(1) - Y(0)| X=x]}
$$

ITE


$$
\displaylines{
D_i = Y_i(1) - Y_0(0) \\
\tau(x) := E[Y(1) - Y(0)| X=x]}
$$



D = (Y_i, X_i, W_i)

在实际的业务问题上，我们只观测到了Y 的一个干预下的结果，所以数据上看


Condition1

no hidden confounder
$$
(\epsilon(0),\epsilon(1)) \perp W |X
$$
Condition2

倾向分有界(为了后面收敛性证明)
$$
0 \le e_{min} \le e(x) \le e_{max} <1
$$




评估指标为estimators with a small Expected Mean Squared Error (EMSE) for estimating the CATE
$$
EMSE(P, \hat \tau) = E[(\tau(X) - \hat\tau(X))^2]
$$




## 解决方案

### Pipeline

三个Stages

1. 估算Control 和Treatment 的response 结果

$$
\mu_0(x) = E[Y (0)|X = x] \\
\mu_1(x) = E[Y (1)|X = x]
$$





2. 利用上述估算结果结合正式的单边真实值，计算两个角度的估算ATE(Imputed Treatment Effect)


$$
D_i^1
:= Y_i^1 − \mu_0(X_i^1) \\
D_i^0
:= \mu_1(X_i^0
) − Y_i^0
$$




3. 对上述的两个估算ATE 进行模型预估，并使用倾向分进行加权。


$$
\tau(x) = g(x)\tau_0(x) + (1 − g(x))\tau_1(x)
$$


### Intuitive Behind

因为有倾向分，所以对于数据不平衡有很好的鲁邦效果





## 效果分析



### 评测指标

采用EMSE 来作为评测指标
$$
EMSE(P, \hat \tau) = E[(\tau(X) - \hat\tau(X))^2]
$$


### 模拟方式

在数据层面，模拟数据考虑了不同的条件， 包括什么情况下S-Learner表现最优，什么情况下T-Learner 最优。 

1. 如果所有个体的treatment effect 是zero 的情况（对S-Learner 会更好）
2. 如果所有个体的treatment effect 都不同
3. 考虑有confounding 和没有confounding 的情况
4. 考虑了treatment 和control 样本数据相同和不同的情况。



在模型层面，还比较了三种不同的Learner 对于基模型采用RF 和BART 的效果差异。实现了树结构和叶子节点独立的RF版本，用R语言实现的Honest RF，称为HTE。在基于BART上，使用dbarts 实现作为基learner



###  模拟结果

比较各种Learner 有两个明显的结论

1. 证明了提出的Meta-Learner 结构不受base learner 的限制，各种都可适配，同时不同的base learner 产生不同的效果
2.  in Simulations 2 and 4 the response functions are globally linear, and we observe that estimators that act globally such as BART have a significant advantage in these situations or when the data set is small. If, however, there is no global structure or when the data set is large, then more local estimators such as RF seem to have an advantage



对于S-Learner 而言，由于其严重依赖Treatment indicator， 像lasso和RFs这样的算法， 可以完全忽略治疗分配由不 选择/分裂它。对于数据模拟其0 treatment effect 是有利的。 但是对于不是0 的情况的数据，则是效果很差。

对于T-Learner 相反的理论，不使用一个模型训练treatment 和control 。 如果是treatment effect 很简单的情况下，T-Learner的两个子模型没有充分的捕捉相似的信号，会效果很差。 但是如果是treatment effect 比较复杂的情况下，u_0 和u_1 没有太多简单的直接联系，那么这时候就会比较效果好。

对于X-Learner而言， 当分配的数据不平衡时候，或者CATE 是结构化的假设的时候，效果就会比以上两种case 都要好，但是如果CATE 是0 的时候，依然效果是不如S-Learner但是比T-Learner 要好。 当treatment effect 形式复杂的时候，效果是最好的。所以当CATE 不是0 的情况下，X-Learner 是最好的方法。 从经验角度看， 小数据量应该是使用BART（Bayesian Additive Regression Trees ） 作为基模型，大数据量使用RF。



### 收敛速率分析

Comparison of Convergence Rates

在这篇文章中，X-Learner 已经证明了效果超过T-Learner 在点估计方向（pointwise estimation rate)

这些结果证明我们的直觉，当Treatment 不平衡时，或者CATE 比Response 的函数形式更加简单的时候，直觉认为X-Learner 是比T-Learner更好。

首先回顾关于minmax nonparametric regression estimation（https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/minimax.pdf）



对于从P 分布产生的N个独立的(X,Y) 的样本，评估预估的估计器的error 从EMSE来评估， 对于一个固定的P，总是会存在有很小EMSE 的估计器。 例如用样本均值来作为预估值，肯定是无偏的。
$$
EMSE(P,\hat \mu_N) = E[(\mu_N(\chi) - \mu_N(\chi))^2]
$$




Lipschitz continuous function
$$
||\nabla f(x) -\nabla f(y)|| \le k *||x-y||
$$




Lipschitz连续，要求函数图像的曲线上任意两点连线的斜率一致有界，就是任意的斜率都小于同一个常数，这个常数就是Lipschitz常数。 所以总体来说，Lipschitz连续的函数是比连续函数较更加“光滑”，但不一定是处处光滑的，比如|X|.但不光滑的点不多，放在一起是一个零测集，所以他是几乎处处的光滑的。



以陆地为举例连续。

岛屿：不连续

一般陆地：连续

丘陵：李普希兹连续

悬崖：非李普希兹连续

山包：可导

平原：线性

半岛：非凸

亚连续（semi-continuity) : 瀑布







## 证明和推导

Definition 1（Families with bounded minmax rate)

For $a \in (0,1]$ , 定义$S(a)$  to be the set of all families F



Definition 2 (Superpopulations with given rates)

回忆一个Superpopulations 的特征



Theorem 1（Minmax rates of the T-Learner)

对于一个分布 
$$
P \in S(a_\mu, a_\tau)
$$


存在一个基础的learner 使得T-Learner 在CATE 估计的速率保持
$$
O(m^{-a_\mu} + n^{-a_\mu})
$$






## 思考



