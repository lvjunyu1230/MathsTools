import numpy as np
import matplotlib.pyplot as plt
import math

# 计算韦伯分布90%置信区间

# 伽马函数
def gamma(x):
    return math.gamma(x)

# 计算weibull分布均值
def weibull_mean(beta, eta):
    return eta * gamma(1 + 1 / beta)

# 计算weibull分布中位数
def weibull_median(beta, eta):
    return eta * (-math.log(2)) ** (1 / beta)

# 计算正态分布90%置信区间
def normal_90_confidence_interval(mean, std):
    return mean - 1.645 * std, mean + 1.645 * std

# 计算韦伯分布90%置信区间
def weibull_90_confidence_interval(beta, eta):
    mean = weibull_mean(beta, eta)
    median = weibull_median(beta, eta)
    std = (mean - median) / 1.645
    return normal_90_confidence_interval(mean, std)

#print(weibull_90_confidence_interval(10.416, 71.8))

# 计算韦伯分布95%置信区间
def weibull_95_confidence_interval(beta, eta):
    mean = weibull_mean(beta, eta)
    median = weibull_median(beta, eta)
    std = (mean - median) / 1.96
    return normal_90_confidence_interval(mean, std)

# 计算正态分布下分数50到分数60的可能性
def normal_probability_between(lower_bound, upper_bound, mean=0, std=1):
    return normal_cdf(upper_bound, mean, std) - normal_cdf(lower_bound, mean, std)

def normal_cdf(x, mean=0, std=1):
    return (1 + math.erf((x - mean) / math.sqrt(2) / std)) / 2

# 计算韦伯分布下分数50到分数60的可能性
def weibull_probability_between(lower_bound, upper_bound, beta, eta):
    return weibull_cdf(upper_bound, beta, eta) - weibull_cdf(lower_bound, beta, eta)

def weibull_cdf(x, beta, eta):
    return 1 - math.exp(-(x / eta) ** beta)

# print(normal_probability_between(50, 60, 68.5, 7.745))
# print(weibull_probability_between(50, 60, 10.416, 71.8))

# ks检验正态分布
def normal_ks_test(data, mean=0, std=1):
    n = len(data)
    data.sort()
    max = 0
    for i in range(n):
        d1 = normal_cdf(data[i], mean, std)
        d2 = (i + 1) / n
        d = abs(d1 - d2)
        if d > max:
            max = d
    return max
# x1 = [70.9, 87.2,101.7,104.2,106.2, 111.4, 112.6, 116.7, 143.9, 150.9]
# max = normal_ks_test(x1, 110.48, 22.31)
# print(max)

x0 = 0.1, 0.17, 0.115, 0.12, 0.24, 0.16, 0.175
x30 = 0.19, 0.26, 0.165, 0.312, 0.3, 0.24, 0.175, 0.335, 0.185
x45 = 0.115, 0.12, 0.2, 0.125, 0.8, 0.215, 0.8, 0.215, 0.19, 0.135

beta1 = 3.731
eta1 = 0.2
beta2 = 4.307
eta2 = 0.3
beta3 = 1.870
eta3 = 0.3

def weibull_ks_test(data, beta, eta):
    n = len(data)
    data.sort()
    max = 0
    for i in range(n):
        d1 = weibull_cdf(data[i], beta, eta)
        d2 = (i + 1) / n
        d = abs(d1 - d2)
        if d > max:
            max = d
    return max

def weibull_pdf(x, beta, eta):
    return beta / eta * (x / eta) ** (beta - 1) * math.exp(-(x / eta) ** beta)

# 给出两个参数， 画出韦伯分布图
def weibull_plot(beta, eta):
    x = np.linspace(0, 1, 100)
    y = [weibull_pdf(i, beta, eta) for i in x]
    plt.plot(x, y)
    plt.show()

# 将三个weibull分布画在一张图上
def weibull_plot_3(beta1, eta1, beta2, eta2, beta3, eta3):
    x = np.linspace(0, 1, 100)
    y1 = [weibull_pdf(i, beta1, eta1) for i in x]
    y2 = [weibull_pdf(i, beta2, eta2) for i in x]
    y3 = [weibull_pdf(i, beta3, eta3) for i in x]
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()

weibull_plot_3(beta1, eta1, beta2, eta2, beta3, eta3)
# 将三个weibull分布画在一张图上，并标注参数
def weibull_plot_3_with_param(beta1, eta1, beta2, eta2, beta3, eta3):
    x = np.linspace(0, 1, 100)
    y1 = [weibull_pdf(i, beta1, eta1) for i in x]
    y2 = [weibull_pdf(i, beta2, eta2) for i in x]
    y3 = [weibull_pdf(i, beta3, eta3) for i in x]
    plt.plot(x, y1, label='data0: beta1=%s, eta1=%s' % (beta1, eta1))
    plt.plot(x, y2, label='data30: beta2=%s, eta2=%s' % (beta2, eta2))
    plt.plot(x, y3, label='data45: beta3=%s, eta3=%s' % (beta3, eta3))
    # 起标题
    plt.title('Weibull Distribution')
    # 画出方格
    plt.grid(True)
    plt.legend()
    plt.show()

weibull_plot_3_with_param(beta1, eta1, beta2, eta2, beta3, eta3)

# 画正态分布图，给出均值和标准差
def normal_plot(mean, std):
    x = np.linspace(0, 1, 100)
    y = [normal_pdf(i, mean, std) for i in x]
    plt.plot(x, y)
    plt.show()
def normal_pdf(x, mean, std):
    return math.exp(-(x - mean) ** 2 / 2 / std ** 2) / math.sqrt(2 * math.pi) / std

# 求正态分布的均值和标准差
def normal_mean_std(data):
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum([(i - mean) ** 2 for i in data]) / n)
    return mean, std

# 将x0, x30, x45三个作正态分布图，画在一张图上，并标注参数
def normal_plot_3(x0, x30, x45):
    mean0, std0 = normal_mean_std(x0)
    mean30, std30 = normal_mean_std(x30)
    mean45, std45 = normal_mean_std(x45)
    x = np.linspace(0, 1, 100)
    y0 = [normal_pdf(i, mean0, std0) for i in x]
    y30 = [normal_pdf(i, mean30, std30) for i in x]
    y45 = [normal_pdf(i, mean45, std45) for i in x]
    plt.plot(x, y0)
    plt.plot(x, y30)
    plt.plot(x, y45)
    # 标注参数
    plt.title('Normal Distribution')
    plt.grid(True)
    plt.legend()
   # 标注区分曲线，数据保留两位小数
    #plt.legend(['data0: mean0=%s, std0=%s' % (round(mean0, 2), round(std0, 2)),
    plt.legend(['data0: mean0=%s, std0=%s' % (round(mean0, 2), round(std0, 2)), 'data30: mean30=%s, std30=%s' % (round(mean30, 2), round(std30, 2)), 'data45: mean45=%s, std45=%s' % (round(mean45, 2), round(std45, 2))])
    plt.show()

# mean0, std0 = normal_mean_std(x0)
# mean30, std30 = normal_mean_std(x30)
# mean45, std45 = normal_mean_std(x45)
# 分别画出三个正态分布图，标注参数

# 矩阵的Jordan标准型
def jordan_normal_form(A):
    # 求特征值和特征向量
    eig_value, eig_vector = np.linalg.eig(A)
    # 求特征值的对角矩阵
    eig_value_diag = np.diag(eig_value)
    # 求特征向量的逆矩阵
    eig_vector_inv = np.linalg.inv(eig_vector)
    # 求Jordan标准型
    J = np.dot(np.dot(eig_vector, eig_value_diag), eig_vector_inv)
    return J

A = np.array([[-2, 1, 0], [-2, 1, -1], [-1, 1, -2]])
J = jordan_normal_form(A)
print(J)

# 矩阵的Jordan标准型，有重根
def jordan_normal_form_with_repeated_roots(A):
    # 求特征值和特征向量
    eig_value, eig_vector = np.linalg.eig(A)
    # 求特征值的对角矩阵
    eig_value_diag = np.diag(eig_value)
    # 求特征向量的逆矩阵
    eig_vector_inv = np.linalg.inv(eig_vector)
    # 求Jordan标准型
    J = np.dot(np.dot(eig_vector, eig_value_diag), eig_vector_inv)
    return J