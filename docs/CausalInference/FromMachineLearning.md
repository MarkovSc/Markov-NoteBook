

## Problem Definition

采用Neyman-Rubin 的潜在结果的模型框架(POF)，并假设一个超总体上的分布 P上产出N 个独立的样本。 X 表示协变量，Y表示两种干预下的潜在结果，T表示干预的indicator。
$$
\displaylines{
(Y_i(0),Y_i(1),X_i,T_i) \sim P \\
ATE := E[Y(1) - Y(0)] \\
\mu_0(X) := E[Y(0)|X=x]\text{    and   } \mu_1(X) := E[Y(0)|X=x]
}
$$


ITE or CATE


$$
\displaylines{
\tau(x) := E[Y(1) - Y(0)| X=x]}
$$



 ### DGP 假设

The model makes the following structural equation assumptions on the data generating process


$$
\begin{split}Y =~& \theta(X) \cdot T + g(X, T) + \epsilon ~~~&~~~ E[\epsilon | X, T] = 0 \\
T =~& f(X, T) + \eta & E[\eta \mid X, T] = 0 \\
~& E[\eta \cdot \epsilon | X, T] = 0\end{split}
$$




## Challenages Or Bias in Causal Inference

那其实最核心的问题就是，为什么某些方法失效， 我们解决的这个问题本身有哪些困难点，哪些误差点， 我们怎么去解决。



建模思路的问题

1.  为什么普通的建模不能做，类似indirect estimation，direction estimation 中的transform等
2.  为什么要将CI 结合ML 进行？
3.  为什么要引入PLR model，而且还保留着D， PLR 的假设有什么优越性
4.  为什么将PLR model 改成残差的形态，然后进行两步建模
5.  到底这里的DML方法有什么优点在对PLR 中的
6.  各种因果推断的树方法有啥差异
7.  正则偏差（ 两个机器学习模型的bias 的和，和两个机器学习bias 的积）



there is no other source of difference between treatment and untreated other than the treatment itself

> The key observation for understanding this phenomena is that g₀(Z)≠𝔼[Y|Z]. Thus, it is not possible in general to get a good estimate of g₀(Z) by “regressing” Y on Z, and this in turn leads to the impossibility of getting a good estimate of θ₀. It is perfectly possible, nevertheless, to get very good predictions of Y given Z and D. This is why we usually say that Machine Learning is good for prediction, but bad for causal inference. 



#### 4.3.2 Native Approach  From Machine Learning

这里有两个参数， target parameter theta, nuisence paramter g

可以使用iterative method 的方法使用随机森林去估计g_0 ,然后使用ols 去回归 $\theta_0$

从OLS 的方式看 那么如果是对$\theta_0$ 进行求解，直接可以得到
$$
argmin_{\theta_{0}} [Y - D\hat \theta_0 - g(x)]^2
$$

$$
2*D * [Y - D\hat\theta_0 -g(x)] = 0 = DY - D^2\hat\theta_0 - Dg(x) \\
then \\
\hat \theta_0 = [D^2]^{-1}D(Y - \hat g(x)) \\
\hat g(x) = Y - D\hat \theta_0 \\
$$

但是这里还有一个$$g_0(x) $$ 也是一个随机变量，并且和x 相关， 所以需要将g_0(x) 先求解出来, 所以这个g(x) 和$\theta$  都是一个随机变量， 如果从MSE 优化的角度是可以直接进行推导的， 问题就是无偏性的保证，和渐进一致性的保证了。

要证明Native Approach 的估计的有偏
$$
\begin{aligned}
\hat \theta = [D^2]^{-1}D(Y - \hat g(x))  &= [D^2]^{-1}D(D\theta_0(X) + g(x) +U - \hat g(x)) \\
&= \theta_0   +  [D^2]^{-1}D(g(x) +U - \hat g(x)) \\
&=\left( \frac{1}{N}\sum_{i=1}^N D_i^2\right)^{-1}\frac{1}{n}\sum D_i(Y_i - \hat g_0(X_i))\\
\end{aligned}
$$

so

$$
\begin{aligned}
\hat\theta - \theta \\ &= [D^2]^{-1}[(g(x) +U - \hat g(x))] \\&= [D^2]^{-1}DU + [D^2]^{-1}D(g(x)- \hat g_0(x))\\ &=
\left( \frac{1}{N}\sum_{i=1}^N D_i^2\right)^{-1} \frac{1}{n}DU_i + \left( \frac{1}{N}\sum_{i=1}^N D_i^2\right)^{-1}\sum_{i=1}\frac{1}{n}D_i(g_0(x_i) - \hat g_0(x_i)) \\
&= a + b \\
&=
 (ED^2)^{-1}\frac{1}{n}DU_i + (ED^2)^{-1}\sum_{i=1}\frac{1}{n}m_i(g_0(x_i) - \hat g_0(x_i)) +(ED^2)^{-1}\sum_{i=1}\frac{1}{n}v_i(g_0(x_i) - \hat g_0(x_i)) \\
 & =a^* + b^* + c^*
\end{aligned}
$$

so
$$
E[\hat\theta - \theta]  = [D^2]^{-1}E[(g(x) +U - \hat g(x))]  = [D^2]^{-1}E[(g(x) - \hat g(x))] \\
Var[\hat\theta - \theta]  = ([D^2]^{-1})^2Var[(g(x) +U - \hat g(x))]
$$

所以这里是否无偏，取决于g(x) 的预估是否无偏， 如果g(x) 是预估无偏的，那么$\theta$ 也是预估无偏的， 那么这里的g(x) 第一步求出来是肯定是有偏差的， 是否要反复收敛， 那么这里会出现问题，这里interest 变量估计的无偏性和渐进一致性都是依赖于nuseriance 的参数的估计的性质。 那如何能够确定这个变量估计是对的呢。

**如果是使用OLS 求解，整体的误差全在g(x) 上，如果g(x) 的求解是有偏的，那么白瞎了，误差只会扩大**。



我们看C Term 也可以发现

<img src="../../../Paper/知识总结/因果推断/Causal_ml.assets/image-20210618211942080.png" alt="image-20210618211942080" style="zoom:30%;" />



 

**Challenges  In Double Machine Learning**

从这个native approach 的角度证明下，直接进行估计是存在偏差的。在这个DGP 和问题假设下， 要得到一个无偏估计量，需要处理biasd  的g(x) 。 实际上这些偏差总结起来就是两种来源。高维复杂模型下的正则损失， 以及过拟合。  例如上述的B 误差对应复杂模型g(x) 下的过拟合，从而导致误差被放大。

The bias has two sources, regularization and overfitting.

- Bias from regularization

  to avoid overfitting the data with complex functional forms, ML use regularization. This decrease the variance of the estimator and reduces overfitting but introduces bias and prevents sqrt n consistency

  实际对应哪个？

- Bias from overfitting
  sometimes our efforts to regularize fail to prevent overfitting( mistaking noise for signal
   	bias and slow convergence )



解决的方案

regularization bias by means of orthogonalization 

overfitting bias by means of cross-fitting. 