import numpy as np
import scipy.stats as sps
import pandas as pd
from tqdm.auto import tqdm

from statsmodels.sandbox.stats.multicomp import multipletests

from collections import defaultdict
from functools import partial
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors
import mpl_toolkits.mplot3d as plt3d
import seaborn as sns

from abc import ABC, abstractmethod 

class ExperimentHandler:
    
    """
    Класс, осуществляющий эксперименты из описания выше. 
    Интерфейс состоит из одной функции run, которая возвращает
    p-значения до и после коррекции (каждым из способов), а также
    количественные меры качества классификации — FDR, FWER и мощность.
    """
    
    def __init__(self, theta_list, Sigma, cov_mx_description, 
                 alpha, sample_size, n_runs, random_seed,
                 criterion, correction_methods, 
                 compute_fwer, compute_fdr, compute_power):
        
        """
        :param theta_list: \theta_0 и \theta_1 из описания
        :param Sigma: матрица ковариаций
        :param cov_mx_description: её описание
        :param alpha: уровень значимости
        :param n_runs: число итераций эксперимента
        :param sample_size: размер выборки на каждом шаге генерации
        :param random_seed: для воспроизводимости результатов
        :param correction_methods: используемые методы МПГ-коррекции
        :param criterion: используемый статистический тест
        :param compute_X: функция, которая вычисляет FWER | FDR | мощность
        """
        
        self.theta_list = theta_list
        self.Sigma = Sigma
        self.cov_mx_description = cov_mx_description
        
        self.alpha = alpha
        self.n_runs = n_runs
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        self.correction_methods = correction_methods
        self.criterion = criterion
        self._compute_fwer = compute_fwer
        self._compute_fdr = compute_fdr
        self._compute_power = compute_power

    def _theta_from_config(self, alt_mask):
        """
        Генерирует вектор средних theta по известной маске
        alt_mask, которая кодирует собой конфигурацию: если
        на позиции i стоит True, то эта компонента берётся
        из theta_1, иначе из theta_0.
        """
        
        theta = self.theta_list[0].copy()
        theta[alt_mask] = self.theta_list[1][alt_mask]
        return theta

    def _sample(self, mean_vec, shape):
        """
        Семплировать случайные вектора из N(mean_vec, Sigma)
        для всех экспериментов сразу. Т.е. на выходе должен
        получиться тензор размерностей, заданных аргументом shape.
        :return: тензор выборок и распределение, из которого они брались
        """
        
        rv = sps.multivariate_normal(
            mean=mean_vec,
            cov=self.Sigma,
            allow_singular=True,
            seed=self.random_seed
        )
        samples = rv.rvs(size=shape)

        assert samples.shape == (*np.atleast_1d(shape), 3), \
            "Некорректный размер тензора samples." \
            f"Нужно {(*np.atleast_1d(shape), 3)}, " \
            f"а пытаются вернуть {samples.shape}"

        return samples, rv

    def _test_hypotheses(self, samples):
        """
        :param samples: Выборки, сгенеррованные в данном эксперименте
                        при какой-то конкретной конфигурации.
                        3D-тензор размерностей (n_runs, sample_size, 3).
        :return: pvalues, матрица p-значений для всех трёх гипотез сразу.
                 Имеет размерность (n_runs, 3)
        """
        
        assert samples.shape == (self.n_runs, self.sample_size, 3), \
            "Некорректный размер тензора samples." \
            f"Нужно {(self.n_runs, self.sample_size, 3)}, " \
            f"пытаются передать {samples.shape}"

        pvalues = self.criterion(
            samples,
            theta_0=self.theta_list[0],
            Sigma=self.Sigma
        )

        assert np.all((0 <= pvalues) & (pvalues <= 1)), \
            "Какие-то из p-значений не лежат в диапазоне [0, 1]"
        assert pvalues.shape == (self.n_runs, 3), \
            "Некорректный размер матрицы pvalues." \
            f"Нужно {(self.n_runs, 3)}, а по факту {pvalues.shape}"

        return pvalues

    def _plot_density(self, alt_mask, ax, n_pts=1000):
        """
        Отрисовать 3D-график с семплированными точками,
        окрашенными согласно теоретической плотности.
        """
        
        samples, rv = {}, {}
        theta_0 = self.theta_list[0]
        samples["theta_0"], rv["theta_0"] = self._sample(theta_0, shape=n_pts)

        theta = self._theta_from_config(alt_mask)
        samples["theta"], rv["theta"] = self._sample(theta, shape=n_pts)

        colors = {"theta_0": "#00CC66", "theta": "#FF3300"}

        for mean_annot in ["theta_0", "theta"]:
            curr_sample = samples[mean_annot]
            ax.scatter(
                curr_sample[:, 0],
                curr_sample[:, 1],
                curr_sample[:, 2],
                c=colors[mean_annot],
                alpha=0.1
            )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    def _plot_marginal_densities(self, alt_mask, ax, cmap, n_pts=1000):

        sns.set(font_scale=1.5)
        h0_samples, h0_rv = self._sample(self.theta_list[0], shape=n_pts)
        theta = self._theta_from_config(alt_mask)
        samples, rv = self._sample(theta, shape=n_pts)

        samples_df = pd.DataFrame(np.ravel(samples),
                                  columns=["Значение"])
        samples_df["Распределение"] = np.full(
            samples_df.shape[0],
            r"$\mathcal{N}(\theta, \Sigma)$"
        )
        h0_samples_df = pd.DataFrame(np.ravel(h0_samples),
                                     columns=["Значение"])
        h0_samples_df["Распределение"] = np.full(
            h0_samples_df.shape[0],
            r"$\mathcal{N}(\theta_0, \Sigma)$"
        )
        joint_samples_df = pd.concat([h0_samples_df, samples_df])
        joint_samples_df["Компонента"] = np.tile(list("123"), 2 * n_pts)
        sns.violinplot(
            x="Компонента",
            y="Значение",
            hue="Распределение",
            ax=ax,
            data=joint_samples_df,
            palette=cmap,
            split=True,
            scale="count",
            inner="quartile",
        )   
     
    def _compute_measures(self, pvalues, alt_mask, suffix):
        """
        Подсчитать FDR, FWER и мощностью.
        :param pvalues: матрица размера (n_runs, 3)
        :param alt_mask: маска, помечающая компоненты, для которых
        основная гипотеза точно отвергается
        :param suffix:
        """

        return {
            f"fdr_{suffix}": self._compute_fdr(pvalues, alt_mask),
            f"fwer_{suffix}": self._compute_fwer(pvalues, alt_mask),
            f"power_{suffix}": self._compute_power(pvalues, alt_mask)
        }

    def _adjust_pvalues(self, pvalues, method):
        """ Скорректировать p-значения заданным методом. """

        reject, adjusted = method(pvalues)
        return reject, adjusted

    def _plot_measures(self, suptitle, titles, measures):
        """ Отрисовывает heatmap с метриками (таблицы из условия) """

        yticklabels = [r"$\mathbf{H_{1}}\colon$ Да",
                       r"$\mathbf{H_{1}}\colon$ Нет"]
        xticklabels = [
            r"$\mathbf{H_{2}}\colon$ Да""\n"r"$\mathbf{H_{3}}\colon$ Да",
            r"$\mathbf{H_{2}}\colon$ Да""\n"r"$\mathbf{H_{3}}\colon$ Нет",
            r"$\mathbf{H_{2}}\colon$ Нет""\n"r"$\mathbf{H_{3}}\colon$ Да",
            r"$\mathbf{H_{2}}\colon$ Нет""\n"r"$\mathbf{H_{3}}\colon$ Нет"
        ]

        sns.set(font_scale=1.7)
        fig = plt.figure(figsize=(20, 3))
        fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=1.07)
        n_panes = len(measures)
        for i in range(n_panes):
            ax = fig.add_subplot(1, n_panes, 1 + i)
            title = titles[i]
            ax.set_title(titles[i], fontsize=16, fontweight="bold")
            annot = measures[i]
            if titles[i] == "Мощность":
                measure = measures[i]
                cmap = sns.diverging_palette(10, 130, n=10)
                transparency = 0.9
            else:
                EPS = 1 / np.sqrt(self.n_runs)
                binarized_measure = np.zeros_like(measures[i])
                above_threshold_mask = (measures[i] > self.alpha + EPS)
                binarized_measure[above_threshold_mask] = 1
                measure = binarized_measure
                cmap = sns.diverging_palette(130, 10, n=10)
                transparency = 0.5

            g = sns.heatmap(
                measure,
                cmap=cmap,
                ax=ax,
                vmin=0, vmax=1,
                annot=annot,
                annot_kws={"color": "black",
                           "weight": "bold"},
                cbar=False,
                alpha=transparency

            )
            ax.set_xticklabels(xticklabels, fontsize=23 - 3 * len(measures))
            ax.set_yticklabels(yticklabels, fontsize=17)
            bottom, top = ax.get_ylim()
            #ax.set_ylim(bottom + 0.5, top - 0.5)

    def _process_configuration(self, config_tuple, scatter_fig, marginal_fig):
        """
        Принимает на вход конфигурацию — какие компоненты \theta
        берутся из \theta_0, а какие из \theta_1. На основании этого
        семплирует вектора, проверяет гипотезы, корректирует их,
        подсчитывает метрики и возвращает всё это в виде словаря.
        """

        i1, i2, i3 = config_tuple
        alt_mask = np.asarray(config_tuple) == 1
        theta = self._theta_from_config(alt_mask)
        samples, rv = self._sample(theta, shape=(self.n_runs,
                                                 self.sample_size))

        sns.set(font_scale=1.4, style='white')
        scatter_ax = scatter_fig.add_subplot(
            2, 4, 1 + i1 * 4 + i2 * 2 + i3,
            projection="3d"
        )
        mask_to_meaning = {0: "Нет", 1: "Да"}
        config_title = (
                r"$\mathbf{H_{1}}\colon$ " + f"{mask_to_meaning[1 - i1]},   "
                + r"$\mathbf{H_{2}}\colon$ " + f"{mask_to_meaning[1 - i2]},   "
                + r"$\mathbf{H_{3}}\colon$ " + f"{mask_to_meaning[1 - i3]}"
        )
        scatter_ax.set_title(config_title, fontsize=16)

        theta = np.array([self.theta_list[i1][0],
                          self.theta_list[i2][1],
                          self.theta_list[i3][2]])
        self._plot_density(alt_mask, scatter_ax)

        sns.set(font_scale=1.4, style='white')
        marginal_ax = marginal_fig.add_subplot(
            2, 4, 1 + i1 * 4 + i2 * 2 + i3
        )
        marginal_ax.set_title(config_title, fontsize=22)
        self._plot_marginal_densities(
            alt_mask,
            ax=marginal_ax,
            cmap="Set2"
        )
        if (i2 != 0) or (i3 != 0):
            marginal_ax.set_ylabel("")
        if i1 == 0:
            marginal_ax.set_xlabel("")
        marginal_ax.legend([])

        results = {"pvalues_raw": self._test_hypotheses(samples)}
        results["reject_raw"] = results["pvalues_raw"] <= self.alpha
        results.update(self._compute_measures(
            results["reject_raw"],
            alt_mask,
            suffix="raw"
        ))

        for method in self.correction_methods:
            results[f"reject_{method}"] = self._adjust_pvalues(
                results[f"pvalues_raw"],
                method=method
            )[0]
            results.update(self._compute_measures(
                results[f"reject_{method}"],
                alt_mask,
                suffix=method
            ))

        return results

    def run(self):
        """
        Запускает симуляции и отрисовывает результаты в виде графиков:
        1. семплированные точки и точки из H_0, окрашенные согласно плотности;
        2. метрики в формате heatmap-ов, без коррекции и с ней (для каждого
           вида коррекции — свой график)
        Также возвращает словарь с результатами эксперимента (p-значения
        и метрики, с ними ассоциированные)
        """

        sns.set(font_scale=1.4, style='white')

        scatter_fig = plt.figure(figsize=(20, 10))
        scatter_fig.suptitle(
            f"{self.cov_mx_description}.\n"
            r"Сравнение $\mathcal{N}(\theta, \Sigma)$ (красный) "
            r"с $\mathcal{N}(\theta_0, \Sigma)$ (зеленый)"
            f"\nДа = гипотеза верна,   Нет = гипотеза не верна",
            fontsize=32, fontweight="bold", y=1.2
        )

        marginal_fig = plt.figure(figsize=(24, 14))
        marginal_fig.suptitle(
            "Сравнение распределений компонент",
            fontsize=26, fontweight="bold"
        )

        experiment_results = defaultdict(list)
        for i1, i2, i3 in tqdm(
                itertools.product([0, 1], [0, 1], [0, 1]),
                "перебираем тройки верных гипотез"
        ):
            config_results = self._process_configuration(
                (i1, i2, i3), scatter_fig, marginal_fig
            )
            for key, value in config_results.items():
                experiment_results[key].append(value)
        for key, value in experiment_results.items():
            if "pvalues" in key or "reject" in key:
                continue
            experiment_results[key] = np.array(value).reshape((2, 4))

        self._plot_measures(
            suptitle="Без МПГ-коррекции",
            titles=["FDR", "FWER", "Мощность"],
            measures=[
                experiment_results[f"{measure_type}_raw"]
                for measure_type in ["fdr", "fwer", "power"]
            ]
        )

        for method in self.correction_methods:
            try:
                suptitle = str(method)
                if method.controls_for == "FWER":
                    titles = ["FWER"]
                    measures = [experiment_results[f"fwer_{method}"]]
                elif method.controls_for == "FDR":
                    titles = ["FDR"]
                    measures = [experiment_results[f"fdr_{method}"]]
                titles.append("Мощность")
                measures.append(experiment_results[f"power_{method}"])
                self._plot_measures(suptitle, titles, measures)
            except KeyError:
                raise NotImplementedError()

        rownames = [r"$\mathbf{H_{0}^{1}}$", r"$\mathbf{H_{1}^{1}}$"]
        colnames = [r"$\mathbf{H_{0}^{2}}, \mathbf{H_{0}^{3}}$",
                    r"$\mathbf{H_{0}^{2}}, \mathbf{H_{1}^{3}}$",
                    r"$\mathbf{H_{1}^{2}}, \mathbf{H_{0}^{3}}$",
                    r"$\mathbf{H_{1}^{2}}, \mathbf{H_{1}^{3}}$"]
        experiment_results = {
            key: (
                pd.DataFrame(value,
                             columns=colnames,
                             index=rownames)
                if ("pvalues" not in key and "reject" not in key)
                else value
            )
            for key, value in experiment_results.items()
        }
        #scatter_fig.tight_layout()
        return experiment_results

    
# Не трогайте этот класс
class AdjustmentMethodABC(ABC):
    def __init__(self, name, alpha, controls_for):
        assert 0 <= alpha <= 1, "Уровень значимости не лежит в (0, 1)"
        assert controls_for in ["FWER", "FDR"],\
                "МПГ-коррекция контролирует либо FWER, либо FDR"
        self.name = name
        self.alpha = alpha
        self.controls_for = controls_for
        
    def __str__(self):
        return f"{self.controls_for}: {self.name}"
    
    @abstractmethod
    def adjust(self, pvalues):
        pass
    
    def __call__(self, pvalues):
        return self.adjust(pvalues)

class AdjustmentMethodTester:

    """
    Класс для тестирования методов МПГ-коррекции и 
    вспомогательных функций подсчета FWER, FDR и мощности.
    """

    def __init__(self,  correction_methods, method_names, 
                 compute_fwer, compute_fdr, compute_power):

        """
        :param correction_methods: используемые методы МПГ-коррекции
        :param method_names: используемые названия для multipletests
        :param compute_X: функция, которая вычисляет FWER | FDR | мощность
        """

        self.correction_methods = correction_methods
        self.method_names = method_names
        self.compute_fwer = compute_fwer
        self.compute_fdr= compute_fdr
        self.compute_power = compute_power

    def run_tests(self, testing_iters=100, runs_per_test=10):
        """
        Запускает базовые тесты на проверку методов МПГ-коррекции
        и подсчета вспомогательных функций.

        :param testing_iters: количество тестов для метода МПГ-коррекции
        :param runs_per_test: количество запусков(число экспериментов) на один тест
        """

        for i, method in tqdm(enumerate(self.correction_methods), 
                               "перебор методов"):
            print(f"{method.name}. Проверка корректности.")
            for _ in tqdm(range(testing_iters), "итерации проверки"):
                pmx = np.random.rand(runs_per_test, 3)
                our_reject, our_adjusted = method(pmx)
                true_reject = []
                true_adjusted = []
                for j in range(runs_per_test):
                    result = multipletests(
                        pmx[j, :], 
                        method=self.method_names[i]
                    )
                    true_reject.append(result[0])
                    true_adjusted.append(result[1])
                true_reject = np.vstack(true_reject)
                true_adjusted = np.vstack(true_adjusted)
                assert np.allclose(our_adjusted, true_adjusted),\
                    f"{method.name} выдаёт не те p-значения после коррекции"
                assert np.all(our_reject == true_reject),\
                        f"{method.name} отвергает не те гипотезы" 

        reject = np.array([[True, False, True, False], 
                           [False, False, False, False], 
                           [False, True, False, False]])
        alt_mask = np.array([True, False, False, False])

        assert self.compute_fwer(reject, alt_mask) == 2/3, \
            "Функция подсчета FWER выдает неверный ответ"
        assert self.compute_fwer(reject[0:2,], alt_mask) == 0.5, \
            "Функция подсчета FWER выдает неверный ответ"
        assert self.compute_fwer(reject[0::2], alt_mask) == 1, \
            "Функция подсчета FWER выдает неверный ответ"

        assert self.compute_fdr(reject, alt_mask) == 0.5, \
            "Функция подсчета FDR выдает неверный ответ"
        assert self.compute_fdr(reject[0:2,], alt_mask) == 0.25, \
            "Функция подсчета FDR выдает неверный ответ"
        assert self.compute_fdr(reject[0::2], alt_mask) == 0.75, \
            "Функция подсчета FDR выдает неверный ответ"

        assert self.compute_power(reject, alt_mask) == 1/3, \
            "Функция подсчета мощности выдает неверный ответ"
        assert self.compute_power(reject[0:2], alt_mask) == 0.5, \
            "Функция подсчета мощности выдает неверный ответ"
        assert self.compute_power(reject[0::3], alt_mask) == 1, \
            "Функция подсчета мощности выдает неверный ответ"

        print("Все тесты пройдены!")
