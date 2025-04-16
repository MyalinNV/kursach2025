import math
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

# Constants
num_d2s_under_h0 = 10000
num_repeat = 10000
sample_sizes = [2**i for i in range(3, 13)]
significance_level = 0.05
rng = np.random.default_rng(seed=0)

# Mixture distribution class
class Mixture:
    def rvs(self, size, random_state):
        ret = 1.0 * random_state.integers(2, size=size)
        i = ret == 0
        ret[i] = stats.norm.rvs(-4/5, 3/5, ret[i].size, random_state=random_state)
        ret[~i] = stats.norm.rvs(4/5, 3/5, ret[~i].size, random_state=random_state)
        return ret

    def cdf(self, xs):
        return (stats.norm.cdf(xs, -4/5, 3/5) + stats.norm.cdf(xs, 4/5, 3/5)) / 2

    def pdf(self, xs):
        return (stats.norm.pdf(xs, -4/5, 3/5) + stats.norm.pdf(xs, 4/5, 3/5)) / 2

class CauchyNormalMixture:
    def rvs(self, size, random_state):
        ret = 1.0 * random_state.integers(2, size=size)
        i = ret == 0
        ret[i] = stats.cauchy.rvs(loc=0, scale=1, size=ret[i].size, random_state=random_state)
        ret[~i] = stats.norm.rvs(loc=0, scale=1, size=ret[~i].size, random_state=random_state)
        return ret

    def pdf(self, xs):
        return 0.5 * stats.cauchy.pdf(xs, loc=0, scale=1) + 0.5 * stats.norm.pdf(xs, loc=0, scale=1)

    def cdf(self, xs):
        return 0.5 * stats.cauchy.cdf(xs, loc=0, scale=1) + 0.5 * stats.norm.cdf(xs, loc=0, scale=1)

class Trimodal:
    def rvs(self, size, random_state):
        ret = 1.0 * random_state.integers(3, size=size)
        ret[ret == 0] = stats.norm.rvs(-2, 0.9, size=(ret == 0).sum(), random_state=random_state)
        ret[ret == 1] = stats.norm.rvs(0, 0.9, size=(ret == 1).sum(), random_state=random_state)
        ret[ret == 2] = stats.norm.rvs(2, 0.9, size=(ret == 2).sum(), random_state=random_state)
        return ret

    def cdf(self, xs):
        return (
            stats.norm.cdf(xs, -2, 0.9)
            + stats.norm.cdf(xs, 0, 0.9)
            + stats.norm.cdf(xs, 2, 0.9)
        ) / 3

    def pdf(self, xs):
        return (
            stats.norm.pdf(xs, -2, 0.9)
            + stats.norm.pdf(xs, 0, 0.9)
            + stats.norm.pdf(xs, 2, 0.9)
        ) / 3

# Fixed support
loc = -4
scale = 8
target_height = 0.16

shoulder_width = (2 / (target_height * scale)) - 1
c = 0.5 - shoulder_width / 2
d = 0.5 + shoulder_width / 2

# Distributions
dists = {
    "Normal(0, 1)": stats.norm(),
    "Normal (0.2, 1)": stats.norm(0.2, 1),
    "Normal(0, 1.1)": stats.norm(0, 1.1),
    "Trapezoidal": stats.trapezoid(c, d, loc=loc, scale=scale),
    "Mixture": Mixture(),
    "Cauchy(0, 1)": stats.cauchy(),
    "Cauchy(0, 0.5)": stats.cauchy(0, 0.5),
    "Cauchy(0.2, 1)": stats.cauchy(0.2, 1),
    "CauchyNormalMixture": CauchyNormalMixture(),
    "Trimodal": Trimodal()
}

dist_pairs = [
    ["Normal (0.2, 1)", "Normal(0, 1)"],
    ["Cauchy(0, 0.5)", "Normal(0, 1)"],
    ["Cauchy(0.2, 1)", "Cauchy(0, 1)"],
    ["CauchyNormalMixture", "Cauchy(0, 1)"],
    ["CauchyNormalMixture", "Normal(0, 1)"],
    ["Trimodal", "Trapezoidal"]
    # ["Normal(0, 1.1)", "Normal(0, 1)"],
    # ["Trapezoidal", "Normal(0, 1)"],
    # ["Mixture", "Normal(0, 1)"],
    # ["Trapezoidal", "Mixture"],
    # ["Mixture", "Trapezoidal"],
]

# D2 statistic
def d2(xs, cdf, axis=0):
    ys = np.repeat(cdf(np.sort(xs, axis)), 2, axis)
    zs = np.repeat(np.linspace(0, 1, xs.shape[axis] + 1), 2)
    ds = ys - zs[(slice(1, -1),) + (None,) * (xs.ndim - axis - 1)]
    return np.max(ds, axis) - np.min(ds, axis)

# Precompute D2 under H0
uniform = stats.uniform()
d2s_under_h0 = {}
print(f"# Precomputation of D2 under H0 ({num_d2s_under_h0:,} each)")
for sample_size in sample_sizes:
    print(f"Sample size: {sample_size:,} ... ", end="", flush=True)
    xs = uniform.rvs((sample_size, num_d2s_under_h0), random_state=rng)
    d2s_under_h0[sample_size] = np.sort(d2(xs, uniform.cdf))
    print("Done")

# Draw D2 under H0
def draw_d2_under_h0():
    sample_sizes_plot = [2**i for i in range(9, 2, -2)]
    fig = plt.figure(figsize=(8, 3))
    ax = fig.subplots()
    ax.set_xlim(0.5, 2.5)

    a = np.linspace(0.5, 2.5, 1000)
    i = np.arange(1, 10)
    sq = np.outer(a, i) ** 2
    asymp_cdf = 1 - 2 * np.sum((4 * sq - 1) * np.exp(-2 * sq), axis=1)
    asymp_cdf[a < 0.4] = 0
    ax.plot(a, asymp_cdf, color="black", label=r"$n \rightarrow \infty$ (asymptotic)")

    for i, sample_size in enumerate(sample_sizes_plot):
        ax.ecdf(
            d2s_under_h0[sample_size] * math.sqrt(sample_size),
            label=f"$n = {sample_size}$ (empirical)",
            color=str((len(sample_sizes_plot) - i - 1) / (len(sample_sizes_plot) + 1)),
            linestyle=(0, (len(sample_sizes_plot) - i, 1)),
        )
    ax.legend()
    ax.set_xlabel(r"$\sqrt{n}D_2(U_n , U)$")
    fig.tight_layout()
    fig.savefig("d2_under_h0.pdf")

draw_d2_under_h0()

# Plot PDFs
fig = plt.figure(figsize=(12, 6))

def dists_pdf(ax, dists_name):
    for name, linestyle in zip(dists_name, ["-", "--", ":"]):
        xs = np.arange(-4, 4, 0.01)
        ax.plot(xs, dists[name].pdf(xs), label=name, color="black", linestyle=linestyle)
    ax.legend()

# Plot 1: Normal vs Cauchy
dists_pdf(fig.add_subplot(221), ["Normal(0, 1)", "Normal (0.2, 1)", "Cauchy(0, 0.5)"])

# Plot 2: Cauchy variants
dists_pdf(fig.add_subplot(222), ["Cauchy(0, 1)", "Cauchy(0.2, 1)", "CauchyNormalMixture"])

# Plot 3: Mixture vs Trapezoidal vs Trimodal
dists_pdf(fig.add_subplot(223), ["CauchyNormalMixture", "Normal(0, 1)", "Trimodal"])

# Plot 4: Trimodal vs Trapezoidal
dists_pdf(fig.add_subplot(224), ["Trapezoidal", "Trimodal"])
fig.tight_layout()
fig.savefig("dists_pdf.pdf")

# Run experiments
print(f"# The experiment ({num_repeat:,} trial each)")
fig = plt.figure(figsize=(2 * 6.4, 2 * 4.8))

for idx, [dist_name, ref_name] in enumerate(dist_pairs):
    print(f"Sampling distribution: {dist_name}, reference distribution: {ref_name}")
    dist = dists[dist_name]
    ref = dists[ref_name].cdf
    ax = fig.add_subplot(3, 2, idx + 1, xscale="log")
    statistical_powers = pd.DataFrame()

    for sample_size in sample_sizes:
        print(f"  Sample size: {sample_size:,}")
        xs = dist.rvs((sample_size, num_repeat), random_state=rng)

        print("    Kolmogorov-Smirnov test: ", end="", flush=True)
        pvalue = stats.ks_1samp(xs, ref).pvalue
        num_rejected = np.count_nonzero(pvalue < significance_level)
        statistical_powers.loc[sample_size, "Kolmogorov-Smirnov"] = num_rejected / num_repeat
        print(f"rejected {num_rejected:,} out of {num_repeat:,}")

        print("    Cramér-von Mises test: ", end="", flush=True)
        pvalue = stats.cramervonmises(xs, ref).pvalue
        num_rejected = np.count_nonzero(pvalue < significance_level)
        statistical_powers.loc[sample_size, "Cramér-von Mises"] = num_rejected / num_repeat
        print(f"rejected {num_rejected:,} out of {num_repeat:,}")

        print("    OVL-2: ", end="", flush=True)
        pvalue = 1 - np.searchsorted(d2s_under_h0[sample_size], d2(xs, ref)) / num_d2s_under_h0
        num_rejected = np.count_nonzero(pvalue < significance_level)
        statistical_powers.loc[sample_size, "OVL-2"] = num_rejected / num_repeat
        print(f"rejected {num_rejected:,} out of {num_repeat:,}")

    ax.set_title(f"Sampling distribution: {dist_name}, reference: {ref_name}")
    ax.set_ylim(-0.05, 1.05)
    statistical_powers.plot(ax=ax, style=[".-k", "x--k", "+:k"])
    ax.legend()

fig.tight_layout()
fig.savefig("result.pdf")
