#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:50:32 2019

@author: alx
"""

# =============================================================================
# конференция НИР 26.04.2019
#
# Асимметрия информации как фактор концентрации капитала
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sstat
import random as rnd
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF

#sns.set(color_codes=True)

# вывод матрицы на экран в компактной форме
#exec(open('/home/alx/Science/Opera/Python/pp.py').read())

# =============================================================================
# pp
# =============================================================================

def pp(x, dec=4):
    """ печатает матрицу в компактной форме
        подавляя числа в научной форме
    in:
        dec: число десятичных знаков
    """
    with np.printoptions(precision=dec, suppress=True):
        print(x)

# =============================================================================
# plot_ecdf
# =============================================================================

def plot_ecdf(samples, low, high, alpha=0.05, figw=12, figh=10):
    """ график ЭФР
    source:
        http://bjlkeng.github.io/posts/the-empirical-distribution-function/
        The Empirical Distribution Function
    """
    n = len(samples)

    ecdf = ECDF(samples)

    x = np.linspace(low, high, 1000)
    ecdf_x = ecdf(x)

    # ширина полосы
    eps = np.sqrt(1. / (2. * n) * np.log(2. / alpha))

    df = pd.DataFrame(ecdf_x, index=x)
    df['ecdf'] = ecdf_x

    #plt.subplot(pltnum)

    plt.figure(1, figsize=(figw, figh))

    df['ecdf'].plot(label='Эмпирическая ФР')
    df['upper'] = pd.Series(ecdf_x, index=x).apply(lambda x: min(x + eps, 1.))
    df['lower'] = pd.Series(ecdf_x, index=x).apply(lambda x: max(x - eps, 0.))
    plt.fill_between(x, df['upper'], df['lower'],
                     alpha=0.1, label='Доверительная полоса')
    #plt.legend(loc='lower right')
    plt.title('ЭФР (n=%d)' % n)
    plt.grid()


# =============================================================================
# make_yield
# =============================================================================

def make_yield(nb_assets, r_mu_min, r_mu_max, sig_min, gamma):
    """ характеристики доходности: вектор мат. ожиданий и ковар. матрица
        и распределение доходностей
    in:
        nb_assets(int): число активов
        r_mu_min, r_mu_max(float): границы мат. ожидания доходностей
        sig_min(float): минимальное ст. отклонение доходностей
        gamma(float): показатель степени в зависимости ст. откл-я дох-сти
                        от его мат. ожидания
    """

    # собственные значения корреляционной матрицы
    # ~ распределение Дирихле
    cor_eigen = np.sort(nb_assets * sstat.dirichlet.rvs([1] * nb_assets))

    # случайная корреляционная матрица с заданными с.з.
    r_cor = sstat.random_correlation.rvs(cor_eigen[0])

    # распределение мат. ожиданий доходностей
    # обязательно добавляется актив с минимальной доходностью
    # значения упорядочены по возрастанию

    r_mu = np.sort(sstat.uniform.rvs(r_mu_min,
                                     r_mu_max - r_mu_min,
                                     size=nb_assets - 1))
    r_mu = np.concatenate(([r_mu_min], r_mu))

    # ст. отклонения доходностей
    # sig_min + степенная функция от мат. ожиданий доходностей


    r_sig = sig_min + r_mu*((r_mu - r_mu_min)/(r_mu_max - r_mu_min))**gamma

    #plt.plot(r_mu, r_sig, r_mu, r_sig, 'b.')
    #plt.grid()

    # ковариационная матрица доходностей
    r_sig_dia = np.diag(r_sig)
    r_Sigma = r_sig_dia @ r_cor @ r_sig_dia

    r_dist = sstat.multivariate_normal(mean=r_mu, cov=r_Sigma)

    return r_mu, r_Sigma, r_dist


# =============================================================================
# opti_port
# =============================================================================

def opti_port(r_mu, r_Sigma, alp):
    """ оптимизация портфеля инвестора
    in:
        r_mu: вектор мат. ожиданий доходностей
        r_Sigma: ковар. матрица доходностей
        alp: коэффициент неприятия риска (alp=1 --
             приближённая полезность Д. Бернулли)
    out:
        w, Fopt, lamb
    """

    one = np.ones(len(r_mu))

    B = r_Sigma + np.outer(r_mu, r_mu)
    Bi = np.linalg.inv(B)

    # множитель Лагранжа
    lamb = (one @ Bi @ r_mu - alp)/(one @ Bi @ one)

    # оптимальные веса
    w = ((r_mu - lamb * one) @ Bi)/alp

    # оптимальное значение целевой функции
    Fopt = w @ r_mu - alp/2 * (w @ B @ w)

    # проверка формулы для опт. полезности
#    Fopt1 = (r_mu @ Bi @ r_mu - ((one @ Bi @ r_mu) - alp)**2/(one @ Bi @ one))/(2 * alp)
#    print(Fopt - Fopt1)

#    var('a b c alpha')
#    z = a/(1+a) - (c/(1+a) - alpha)^2/(b - c^2/(1+a))
#    z.full_simplify().show()

#    Sigmai = np.linalg.inv(r_Sigma)
#    a = r_mu @ Sigmai @ r_mu
#    b = one @ Sigmai @ one
#    c = r_mu @ Sigmai @ one
#
#    Fopt2 = (a/(1+a) - (c/(1+a) - alp)**2/(b - c**2/(1+a)))/(2*alp)
#    print(Fopt - Fopt2)

    return w, Fopt, lamb


# =============================================================================
# ecdf_port
# =============================================================================

def ecdf_port(r_mu, r_Sigma, r_dist, alp, nb_samp):
    """ ЭФР доходности портфеля
    in:
        nb_samp: число выборок
    """

    # оптимизация портфеля
    w, _, _ = opti_port(r_mu, r_Sigma, alp=alp)
    #pp(w)

    # реализованные доходности активов
    r_rand = r_dist.rvs(size=nb_samp)

    # реализованные дох-сти портфеля
    r_port = r_rand @ w.T

    plt.figure(1, figsize=(12, 10))
    plot_ecdf(r_port, .2, .7)


# =============================================================================
# rnd_subs
# =============================================================================

def rnd_subs(r_mu, r_Sigma, alp, nb_act):
    """ выбирает случайно активы и обсчитывает портфель по ним
    in:
        nb_act: число активов
    """
    d = len(r_mu)
    assert nb_act < d

#    rnd.seed(1234)

    # случайные фрагменты элементов доходностей
    sel_act = rnd.sample(range(d), k=nb_act)
    sel_act.sort()
#    pp(sel_act)

    r_mu1 = r_mu[sel_act]
    r_Sigma1 = r_Sigma.take(sel_act, axis=0).take(sel_act, axis=1)

    w, Fopt, _   = opti_port(r_mu, r_Sigma, alp=alp)
    w1, Fopt1, _ = opti_port(r_mu1, r_Sigma1, alp=alp)
    pp(Fopt - Fopt1)

    # стохастическое доминирование 2-го порядка
    p_mu = w @ r_mu
    p_mu1 = w1 @ r_mu1

    p_sig2 = w @ r_Sigma @ w
    p_sig21 = w1 @ r_Sigma1 @ w1

    print('Стох. доминирование 2-го порядка:')
    pp(p_mu - p_mu1)
    pp(p_sig2 - p_sig21)


# =============================================================================
# stoch_dom2
# =============================================================================

def stoch_dom2(r_mu, r_Sigma, alp):
    """ проверяет условия стох. доминирования 2-го порядка
        при ограничениях;
        переформулировано условие монотонности
        мат. ожидания доходности портфеля
        r_mu()
    """

    assert len(r_mu.shape) == 1

    dim = len(r_mu)

    r_mu = np.reshape(r_mu, (dim, 1))

    one = np.ones((dim, 1))
    e1 = np.zeros((dim, 1))
    e1[0, 0] = 1

    B = r_Sigma + np.outer(r_mu, r_mu)
    Bi = np.linalg.inv(B)
    #print('Bi')
    #pp(Bi)

    a = (one.T @ Bi @ one)[0, 0]
    b = (one.T @ Bi @ e1)[0, 0]
    d = (one.T @ Bi @ r_mu)[0, 0]
    c = (e1.T  @ Bi @ e1)[0, 0]
    e = (e1.T  @ Bi @ r_mu)[0, 0]
    #f = (r_mu.T  @ Bi @ r_mu)[0, 0]

    #pp(np.array([a, b, d, c, e, f]))

    # исходное решение (ww <-> w_wave)
    lambdaw = (d - alp)/a
    ww = ((r_mu - lambdaw * one).T @ Bi)/alp
    alp_ww_mu = (alp * ww @ r_mu)[0, 0]
    ww_Sig_ww = (ww @ r_Sigma @ ww.T)[0, 0]
    print('Исходное решение:')
    print('ww=')
    pp(ww)
    #pp([alp_ww_mu, ww_Sig_ww])
    print()

    # ограниченное решение
    Ri = np.linalg.inv(np.matrix([[a, b], [b, c]]))
    lambdas = Ri @ np.matrix([[d - alp], [e]])
    w = ((r_mu - lambdas[0, 0]*one - lambdas[1, 0]*e1).T @ Bi)/alp
    alp_w_mu = (alp * w @ r_mu)[0, 0]
    w_Sig_w = (w @ r_Sigma @ w.T)[0, 0]
    print('Ограниченное решение:')
    print('w=')
    pp(w)
    #pp([alp_w_mu, w_Sig_w])
    print()

    print('Стох. доминирование 2-го порядка')
    print('w*mu:      ', alp_ww_mu >= alp_w_mu)
    print('w*Sigma*wT:', ww_Sig_ww <= w_Sig_w)

    # условие монотонности по w*mu
    k = b*d - a*e
    print('k/b =', k/b)


# =============================================================================
# test_alpha
# =============================================================================

def test_alpha(r_mu, r_Sigma):
    """ определяет значения k/b для разных позиций
    """

    dim = len(r_mu)

    one = np.ones((dim, 1))

    B = r_Sigma + np.outer(r_mu, r_mu)
    Bi = np.linalg.inv(B)

    idm = np.identity(dim)

    alpha_max = np.zeros(dim)

    for j in range(dim):
        e1 = idm[:, j]

        a = (one.T @ Bi @ one)[0, 0]
        b = (one.T @ Bi @ e1)[0]
        d = (one.T @ Bi @ r_mu)[0]
        e = (e1.T  @ Bi @ r_mu)
        alpha_max[j] = (b*d - a*e)/b

    return alpha_max


# =============================================================================
# hhi
# =============================================================================

def hhi(x, inve=False):
    """ HHI concentration index
    in:
        x: real vector (not normalized)
        inve(bool): return
    """

    assert len(x.shape) == 1

    assert len(x) > 0

    assert all(x >= 0)

    xs = np.sum(x)
    assert xs > 0

    hh = np.sum((x / xs)**2)
    if inve:
        return 1/hh

    return hh


# =============================================================================
# sim_cap
# =============================================================================


# 50 * 365 * 10 = 182500
# капитал в тыс. руб

# =============================================================================
# параметры случайных доходностей
# =============================================================================

np.random.seed(1234)
rmu, rSigma, rdist = make_yield(nb_assets=5,
                                r_mu_min=.03,
                                r_mu_max=.25,
                                sig_min=0.001,
                                gamma=2)

print("r_mu:")
pp(rmu)

print("r_Sigma:")
pp(rSigma, 6)

#r_mu:
#[0.03   0.0997 0.1174 0.1733 0.2035]
#r_Sigma:
#[[ 0.000001  0.000004  0.000004 -0.000016  0.000041]
# [ 0.000004  0.000121 -0.000088  0.000012  0.000218]
# [ 0.000004 -0.000088  0.000381  0.000419  0.000829]
# [-0.000016  0.000012  0.000419  0.005555  0.001057]
# [ 0.000041  0.000218  0.000829  0.001057  0.016284]]


# =============================================================================
# изучение стох. доминирования 2-го порядка
# =============================================================================

# параметры из CRes.xls
#mu= np.array([-1,-2,-1])
#Sig = np.matrix([[2.421056127, 3.450517653, -4.090719899],
#                 [3.450517653, 6.32577729,	-8.221926932],
#                 [-4.090719899, -8.221926932, 11.18502897]])

#stoch_dom2(np.array([1,2,-1]), Sig, alp=3.3994)

#test_alpha(np.array([1,2,-1]), Sig)
#array([ 3.39936113, -1.33092264, -0.15159931])

#test_alpha(np.array([-1,2,-1]), Sig)
#array([ 1.54467442, -0.85398548, -0.19162154])

#test_alpha(np.array([-1,-2,-1]), Sig)
#array([-0.71057663, -2.05313633,  0.64045455])

# стох. доминирование с rmu, rSigma
pp(test_alpha(rmu, rSigma))
#[ 9.6227 32.8661 33.3698 18.0875 49.119 ]

stoch_dom2(rmu, rSigma, alp=9.2)
#Исходное решение:
#ww=
#[[-0.0647  0.7159  0.379  -0.0092 -0.0211]]
#
#Ограниченное решение:
#w=
#[[ 0.      0.6235  0.3739  0.0125 -0.0099]]
#
#Стох. доминирование 2-го порядка
#w*mu:       True
#w*Sigma*wT: True
#k/b = 9.622736701810924

# =============================================================================
# Оптимизация портфеля инвестора
# =============================================================================

pp(opti_port(rmu, rSigma, alp=2))
#(array([-5.3673,  4.2952,  2.2628, -0.0763, -0.1144]),
#0.24826664369392243, 0.0001838625914379371)

# =============================================================================
# генерация доходностей
# =============================================================================

pp(rdist.rvs(size=10))
#[[ 0.0293  0.0902  0.1158  0.2535  0.2194]
# [ 0.0298  0.0863  0.1251  0.1101  0.1526]
# [ 0.0297  0.1153  0.0892  0.2326  0.0987]
# [ 0.0295  0.0954  0.1512  0.2463  0.4392]
# [ 0.028   0.1065  0.086   0.0997 -0.0457]
# [ 0.0305  0.1055  0.1182  0.0895  0.2725]
# [ 0.0306  0.1208  0.0753  0.1242  0.282 ]
# [ 0.0303  0.0982  0.1263  0.24    0.2201]
# [ 0.0289  0.0922  0.1249  0.2502  0.4253]
# [ 0.0303  0.0935  0.1     0.121   0.1512]]

# =============================================================================
# проверка ЭФР
# =============================================================================

plot_ecdf(np.random.uniform(size=1000), -0.5, 1.5)

# =============================================================================
# распределение доходностей портфеля
# =============================================================================

ecdf_port(rmu, rSigma, rdist, 2, 1000)

# =============================================================================
# случайный выбор активов и оптимизация портфеля по ним
# =============================================================================

rnd_subs(rmu, rSigma, 1, 4)


# =============================================================================
# симуляция
# =============================================================================



#%%

def sim_cap(nb_inve,
            r_mu, r_Sigma, r_dist,
            alp,
            cap_min, cap_c,
            acc_coe,
            nb_years,
            pri=True):
    """ симулирует динамику капиталов
    in:
        nb_inve: число инвесторов
        r_mu, r_Sigma, r_dist: параметры распределения дох-стей
        alp: коэфф-т неприятия риска (один и тот же для всех
             инвесторов)
        cap_min: минимальный капитал
        cap_c: параметр начального распределения Парето
        acc_coe: коэффициент логистического распределения,
                 задающего вероятность выбора актива в
                 зависимости от дохода
        nb_years: число лет симуляции
        pri(bool): вывод результатов на экран
    """

    # для того, чтобы существовало мат. ожидание
    assert cap_c > 1

    # начальное распределение
    cap0 = cap_min * (1/np.random.uniform(size=nb_inve))**(1/cap_c)
    #pp(cap0)

    #print('Исходная концентрация капиталов: 1/hhi=', hhi(cap0, True))

    # ЭФР
    #plot_ecdf(cap0, 0.95 * cap_min, 1.05 * np.max(cap0))

    # мат. ожидание начального размера капитала
    cap_expect = cap_min * cap_c / (cap_c - 1)
    if pri:
        print('E[K0]=', cap_expect)

    # число активов
    nb_assets = len(r_mu)

    # матрица размеров капиталов
    cap_mat = np.zeros((nb_inve, nb_years + 1))
    cap_mat[:, 0] = cap0
    if pri:
        print('Капиталы {инвестор x год}:')
        pp(cap_mat, 1)

    for ye in range(nb_years):
        if pri:    
            print('\n----- Год %i -----' % ye)

        # доступ к активам
        access = np.zeros((nb_inve, nb_assets))

        # портфели инвесторов
        w_inve = np.zeros((nb_inve, nb_assets))

        # вероятность доступа к активам
        #acc_prob = 1/(1 + np.exp(-acc_coe * np.log(cap/cap_expect)))
        acc_prob = 1/(1 + (cap_mat[:, ye]/np.mean(cap_mat[:, ye]))**(-acc_coe))

        if False:
            plt.figure(1, figsize=(8, 5))
            plt.plot(cap_mat[:, ye], acc_prob, '.')
            plt.grid()
        
        if pri:
            print('Вероятности доступа к активам {инвестор}:')
            pp(acc_prob, 3)

        # цикл по инвесторам
        for i in range(nb_inve):
            
            # случайный выбор доступных активов
            z = np.random.uniform(size=nb_assets) <= acc_prob[i]
            access[i, z] = 1
            # актив с минимальной доходностью и риском
            # включается всегда
            access[i, 0] = 1

            # оптимизация портфеля инвестора
            sel_act = access[i, :] == 1
            #print(sel_act)

            r_mu1 = r_mu[sel_act]
            r_Sigma1 = r_Sigma[sel_act, :]
            r_Sigma1 = r_Sigma1[:, sel_act]
            #pp(r_Sigma1)

            w1, _, _ = opti_port(r_mu1, r_Sigma1, alp=alp)
            #pp(w1)
            w_inve[i, sel_act] = w1

        if pri:
            print('\nДоступ к активам {инвестор x актив}:')
            pp(access)

            print('\nПортфели {инвестор x актив}:')
            pp(w_inve)

        # реализованные доходности активов
        r_rand = r_dist.rvs(size=1)

        if pri:
            print("\nРеализованные доходности {актив}:")
            pp(r_rand)

        # реализованные доходности портфелей инвесторов
        r_port = w_inve @ r_rand
        
        if pri:
            print("\nРеализованные доходности портфелей {инвестор}:")
            pp(r_port, 3)

        # новые размеры капиталов
        cap_mat[:, ye + 1] = cap_mat[:, ye] * (1 + r_port)

    if pri:
        print('\nКапиталы {инвестор x год}:')
        pp(cap_mat, 1)
    
    # 1/HHI для капиталов
    ihhi = np.zeros(1 + nb_years)
    for ye in range(1 + nb_years):
        ihhi[ye] = hhi(cap_mat[:, ye], True)
    
    if pri:
        print('\n1/HHI {год}')
        pp(ihhi)
    
    #plt.plot(ihhi)
    
    return ihhi

    # это тоже интересный график -- двумерное распределение
    #sns.distplot(cap_mat, hist=False, rug=True);

    plt.figure(2, figsize=(8, 5))
    for ye in range(nb_years + 1):
        sns.distplot(cap_mat[:, ye], hist=False, rug=True, label='Год %i' % ye)
    plt.legend(loc='upper right')

    plot_ecdf(cap_mat[:, 0],
              0.95 * np.min(cap_mat), 1.05 * np.max(cap_mat), figw=8, figh=5)
    plot_ecdf(cap_mat[:, nb_years],
              0.95 * np.min(cap_mat), 1.05 * np.max(cap_mat), figw=8, figh=5)

# число повторений
n_repe = 10
nbyears = 50

ihhi_m = np.zeros((n_repe, 1 + nbyears))
for repe in range(n_repe):
    ihhi_m[repe, :] = sim_cap(10,
          rmu, rSigma, rdist,
          alp=2,
          cap_min=180, cap_c=2,
          acc_coe=3,
          nb_years=nbyears,
          pri=False)

print('\n1/HHI {повторение x год}')    
#pp(ihhi_m)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 22
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 1

with plt.style.context('fast'):
    plt.plot(ihhi_m.T, 'k')
plt.xlabel('Год')
plt.ylim(bottom = 0)
plt.ylabel('1/HHI')
plt.grid()
plt.tight_layout()

plt.savefig('iHHI.svg')
plt.savefig('iHHI.eps')
plt.savefig('iHHI.png')

rmu
pp(rSigma*1e5, 3)
