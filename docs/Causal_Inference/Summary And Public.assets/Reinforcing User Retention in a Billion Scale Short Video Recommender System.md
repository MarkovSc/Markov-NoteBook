# Reinforcing User Retention in a Billion Scale Short Video Recommender System

https://zhuanlan.zhihu.com/p/610023562

缺乏环境做复杂的大模型，

这种方式和常规的Q-Learning 是啥区别

到底做了什么改进？

能否通过这篇文章，看到RelateWork，或者一些统一化的想法



## 1. Backgroup

短视频推荐核心的目标是提升用户的留存，推荐的核心目标更倾向于提升DAU 这样的指标。 留存是一个系统和用户多次交互之后的反馈效果。留存很难被分解归因到一系列行为中的某个或者整体序列。因此传统的point-wise 或者list-wise 的模型很难去优化留存。 在这篇文章中，我们选择强化学习的方法来优化留存，最大化长期的指标。把这个问题定义为基于无限视野的请求粒度的马氏决策的过程（ infinite-horizon request-based markov Decision Process）。同时我们的最大化目标是minimize the accumulated time interval of multiple sessions, which is equal to improving the app open frequency and user retention.

最小化累计多个会话的时间间隔？ -》 提升打开频次 =》留存？

然而强化学习的方法不能够直接的使用在这个场景上，因为uncertainty，bias，and long delay time incurred by 用户留存。  

提出了一种方法Dubbed RLUR 去处理这个问题，



## 2. Challenage and Contribution

Uncertainty

留存指标受其他策略影响较大，受推荐策略效果有其他干预的噪声

Bias

在不同的时间周期有差异，并且高活跃用户贡献高留存

Long Decay

长期的观测才能得出结果





## 2. Problem Definition

问题被建模成为一个无限视野的Markov Decision Process 的过程。







## 3. Solution

用户和app交互提供了短暂的反馈包括watch time and other interactions。 用户离开app，那么session 终止了， 如果用户离开app 了，这个session 就会终止。 我们的目标是最小化

Our objective

is to minimize the cumulative returning time (defined as the

time gap between the last request of the session and the first request

of the next session),













### Retention Critic Learning

就是学习一个critic function Q, 由$w_T$ 参数化的一个根据状态和action 返回的值， 其实和普通的uplift 没什么区别
$$
Q_T(s_{it, a_{it}|w_T}) 
$$
区别就在于这个Q 是一个长期的奖励，需要进行根据长期奖励进行bp，并且只有当前的response ，并没有长期的， 属于无长期Label，只有当前的Label 下想预估长期Label 的uplift 问题。

>Q-Learning
>
>Target Network
>
>Behavior Network
>
>loss = Q(S_t)  - (r_st + 上一次的)



### Methods for Delayed Reward

返回的时间奖励仅仅发生在每个session 的最后一个部分，以延迟的奖励进行RL Policy 的学习是效率很低的，为了更好的理解用户的留存，我们采用了启发式奖励的方式（heuristic reward methods) 去加强policy learning and intrinsic motivation（内在激励） 的方式使得policy 能够探索到新的states。 由于反馈feedback 和returning time 是正相关，我们使用当前的奖励作为启发式的奖励去引导policy 提升用户的留存。

采用RND（random Network Distillation）的方法，初始化两个相同结构的网络，训练一个网络结构去适配另外一个固定网络的输出（fit the output of another fixed network）。
$$
L(w_e) = \sum_{u_{it \in D}} ||E(u_{it})||^2
$$


### Uncertainty

returning time 是非常的不置信，而且容易被系统外部的很多其他因素影响的。为了降低方差，提出了一种zhegn'jai









## Experiments



**离线评估**

基于公开的数据集建立一个模拟器，包括三个部分

- 预测不同用户行为下的立即的反馈（immediate feedback module)
- 预测用户是否离开app(leave module)
- 预测在每个session 之后，用户在第1-k 天返回的概率，k 最大是10



比较了RLUR 方法基于Block-box 优化的调试之后的模型结果，基于CEM 的方法， 和基于TD3的方法。

我们评测了每个方法的平均返回天数（average returning day， returning time） 以及用户的第一天的平均留存率。训练了每个算法直到收敛，





**实时评估**

在快手的平台上进行实时的效果评估，将用户分为两个部分，一个部分采用RLUR 方法，另外一个部分采用CEM， 然后评估两种算法在这两部分人群上的app打开频次，DAU， 用户第一天的留存（次留）。

>留存： 用户到n 天的时候，还在继续使用产品
>
>流失： 用户到n天的时候，还没使用产品



**Inference and Training**



对于每个请求，user state 发送到Actor上，Actor 返回均值和方差。Action 从高斯分布$N(\mu, \sigma)$中产生, 排序函数将action 和每个视频的预测分乘积，推荐6个打分。

<img src="Reinforcing User Retention in a Billion Scale Short Video Recommender System.assets/image-20230420192434926.png" alt="image-20230420192434926" style="zoom:50%;" />







