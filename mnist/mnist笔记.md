# 【图像处理】-024 范数

[toc]

## 1 定义

&emsp;&emsp;**范数**用于衡量一个向量的大小。形式上$L^p$范数的定义如下：
$$
L^{p}=(\sum_{i}|x_i|^p)^\frac{1}{p} \tag{1}
$$
&emsp;&emsp;其中，$p \in R,p \geq 1$.

&emsp;&emsp;范数，包括$L^p$范数，是将向量映射到非负值的函数。

&emsp;&emsp;向量$x$的范数衡量从原点到点$x$的距离。范数是满足以下性质的任意函数
- $f(x) = 0 \Longrightarrow x=0;$
- $f(x+y) \leq f(x) + f(y);$  (**三角不等式**)
- $\forall \alpha \in R, f(\alpha x)=|\alpha|f(x)$


