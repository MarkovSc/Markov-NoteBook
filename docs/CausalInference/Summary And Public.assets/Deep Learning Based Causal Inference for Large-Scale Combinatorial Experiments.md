# Deep Learning Based Causal Inference for Large-Scale Combinatorial Experiments



## Background && Contribution

大规模的在线平台每天都过百的随机实验A/B 实验开启来迭代产品的功能和营销策略。 但是这些干预的组合却缺乏详细的测试。 那这就产生一个工业问题， 如果我们没有观测到所有treatment 的组合的情况下，如何通过现有的treatment 的组合数据来进行预测，并得到最优的组合策略？ 基于这个问题，开发了一个novel 的框架，对于每个个体仅观测到部分treatment 的情况下， 结合神经网络和doubly robust estimation的方式来估计对于每个个体所有的treatment 组合的效果。 这套方法叫DeDL(debiased deep learning) 利用Neyman 正交性并结合神经网络的灵活性和可解释性。  并且证明了这个框架的有效性，和渐进一致性（effecient, consistent and asymptotically normal estimators under mild assumptions)， 最后实际在快手的平台上进行了部署





multiple experiments , multiple treatements, combinatorial experiments



## Solution &&  effectiveness

In Section 3.1, we first introduce the DL framework built upon Farrell et al. (2020) to study the

estimation and inference of treatment effects in our multiple experiments setting 

Leveraging the

semiparametric influence function derived via the pathwise derivative method



DGP
$$
E[Y|X=x, T=t] =  G(\theta^*(x),t)
$$


关键这里的问题是t 也是一个函数结构， $\theta$ 是否和t 相关？





#### Benchmarks



**Linear Addition Approach && Liner Regression Approach**
$$
\mu(t_1 + t_2) = \mu(t_1) + \mu(t_2) \\
\mu(t_1 + t_2) = w_1 * \mu(t_1)  + w_2 *\mu(t_2)
$$
这种线性相加的假设，是基于两个

The

standard error of an LA estimator for any treatment combination is estimated by assuming that

the estimators for individual experiments are independent.

We remark that both the LA and LR approaches

inherently assume that the treatment effects are linearly additive and homogeneous.





**Pure Deep Learning Approach**

empolys a DNN with a similar structure as DeDL that predicts the outcome variables y as a function fo both x and t.

DeDL 的方法是需要有个concrete link function to describe the  relationship of t to y conditional on x





**Structured Deep Learning**




$$

$$




