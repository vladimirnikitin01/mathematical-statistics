import numpy as np
import scipy.stats as sps

from mht import AdjustmentMethodABC


def compute_fwer(reject, alt_mask):
    """
    Функция подсчета FWER.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка fwer по данным экспериментам.
    """

    alt_mask = np.array(alt_mask)
    reject = np.array(reject)
    n, m = reject.shape
    count_errors_first = 0
    for i in range(n):
        v_p_s = np.sum((~alt_mask) * reject[i])
        if v_p_s > 0:
            count_errors_first += 1

    fwer = count_errors_first / n

    assert np.isscalar(fwer)
    assert 0 <= fwer <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return fwer


def compute_fdr(reject, alt_mask):
    """
    Функция подсчета FDR.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка fdr по данным экспериментам.
    """

    alt_mask = np.array(alt_mask)
    reject = np.array(reject)
    n, m = reject.shape
    sum_divisions = 0
    for i in range(n):
        v_p_s = np.sum((~alt_mask) * reject[i])
        r_s = np.sum(reject[i])
        sum_divisions += (v_p_s / np.maximum(1, r_s))
    fdr = sum_divisions / n

    assert np.isscalar(fdr)
    assert 0 <= fdr <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return np.nan_to_num(fdr)


def compute_power(reject, alt_mask):
    """
    Функция подсчета мощности.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка мощности по данным экспериментам.
    """

    # Если данные полностью согласуются с основными гипотезами, то
    # мощность не определена, потому вместо неё возвращаем np.nan
    # (так полезно сделать, чтобы heatmap-ы нормально отрисовались)
    if ~np.any(alt_mask):
        return np.nan

    alt_mask = np.array(alt_mask)
    reject = np.array(reject)
    n, m = reject.shape
    sum_divisions = 0
    for i in range(n):
        v_p_s = np.sum((~alt_mask) * reject[i])
        r_s = np.sum(reject[i])
        m0 = np.sum(~alt_mask)
        sum_divisions += (r_s - v_p_s) / np.maximum(1, m - m0)
    power = sum_divisions / n

    assert np.isscalar(power)
    assert 0 <= power <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return power


class BonferroniAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Бонферрони",
            alpha=alpha,
            controls_for="FWER"
        )
        self.alpha = alpha

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2, \
            "По условию задачи, adjust принимает матрицу " \
            "размерностей (число_экспериментов, количество_гипотез)"

        pvalues = np.array(pvalues)
        n, m = pvalues.shape
        adjusted = np.minimum(np.array(pvalues) * m, 1)
        reject = adjusted <= self.alpha

        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape), \
            "Размерности матриц reject и adjusted " \
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)), \
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


class HolmAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Холма",
            alpha=alpha,
            controls_for="FWER"
        )
        self.alpha = alpha

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2, \
            "По условию задачи, adjust принимает матрицу " \
            "размерностей (число_экспериментов, количество_гипотез)"

        pvalues = np.array(pvalues)
        n, m = pvalues.shape
        # теперь давайте реализуем эти оценки
        adjusted = np.copy(pvalues)
        for experiment in range(n):
            index_sort = np.argsort(pvalues[experiment])
            for i, index in enumerate(index_sort):
                if i == 0:
                    adjusted[experiment, index] = np.minimum(1, pvalues[experiment, index] * (m + 1 - (i + 1)))
                else:
                    max_with_last = np.maximum(pvalues[experiment, index] * (m + 1 - (i + 1)),
                                               adjusted[experiment, index_sort[i - 1]])
                    adjusted[experiment, index] = np.minimum(1, max_with_last)

        reject = adjusted <= self.alpha

        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape), \
            "Размерности матриц reject и adjusted " \
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)), \
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


class BenjaminiYekutieliAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Бенджамини-Иекутиели",
            alpha=alpha,
            controls_for="FDR"
        )
        self.alpha = alpha

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2, \
            "По условию задачи, adjust принимает матрицу " \
            "размерностей (число_экспериментов, количество_гипотез)"

        pvalues = np.array(pvalues)
        n, m = pvalues.shape
        c = np.sum([1 / i for i in range(1, m + 1)])
        # теперь давайте реализуем эти оценки
        adjusted = np.copy(pvalues)
        for experiment in range(n):
            index_sort = np.flip(np.argsort(pvalues[experiment]))
            for i, index in enumerate(index_sort):
                i_inverse=len(index_sort)-i-1 #порядок в отсортирован
                if i == 0:  # то есть с конца p, которые во возр отсортированы
                    adjusted[experiment, index] = np.minimum(1, pvalues[experiment, index] * m*c/(i_inverse+1))
                else:
                    min_with_future = np.minimum(pvalues[experiment, index] * m*c/(i_inverse+1),
                                               adjusted[experiment, index_sort[i - 1]])
                    adjusted[experiment, index] = np.minimum(1, min_with_future)

        reject = adjusted <= self.alpha

        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape), \
            "Размерности матриц reject и adjusted " \
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)), \
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


def criterion(samples, theta_0, Sigma):
    """
    Векторная реализация равномерно наиболее мощного критерия 
    для правосторонней гипотезы из условия.
    
    :param samples: Матрица выборок размера 
                    (число_экспериментов, размер_выборок, размерность_пространства)
    :param theta_0: Вектор средних, соответствующий основным гипотезам. 
                    Размер (размерность_пространства,)
    :param Sigma: Матрица ковариаций компонент вектора. 
                  Размер (размерность_пространства, размерность_пространства)
    
    :return pvalues: Матрица p-value, имеет размер 
                    (число_экспериментов, количество_гипотез).
                    В позиции (k, j) записано p-value критерия S_j для проверки 
                    гипотезы H_j в эксперименте k.
    """
    assert samples.ndim == 3, "На вход должен подаваться 3-мерный тензор"
    n_runs, sample_size, n_hypotheses = samples.shape

    x_mean = np.mean(samples, axis=1)
    res = x_mean - theta_0
    res *= np.sqrt(sample_size)
    res /= np.sqrt(np.array([Sigma[0, 0], Sigma[1, 1], Sigma[2, 2]]))
    pvalues = sps.norm.sf(res)


    assert pvalues.shape == (n_runs, n_hypotheses), \
        "Некорректная форма матрицы p-значений." \
        f"Должно быть {(n_runs, n_hypotheses)}, a вместо этого {pvalues.shape}"
    return pvalues
