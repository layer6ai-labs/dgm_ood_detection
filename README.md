# OOD Detection with Likelihood-based Deep Generative Models

The codebase used to produce the results for ["A Geometric Explanation of the Likelihood OOD Detection Paradox"](https://arxiv.org/abs/2403.18910), accepted to ICML 2024. 

# News: We have revamped the codebase in our new [website](https://layer6ai-labs.github.io/dgm_geometry/sections/ood.html)

<p align="center">
  <img src="./figures/fig2-aria-1.png" alt="Explanation of OOD failure" />
</p>

# Overview

In many likelihood-based generative models, the model specifies a density function over all the possible datapoints, and the training process typically aims to increase the density for training (in-distribution) data. We know that densities integrate to one over all the possible datapoints, and as a consequence, one can reasonably assume that this would also decrease the density regions that are far apart from the training data, i.e., out-of-distribution (OOD) regions. Paradoxically, based on research first presented in Nalisnick et al.'s work titled ["Do deep generative models know what they don't know?"](https://arxiv.org/abs/1810.09136), likelihood values (or probability densities) alone are not a reliable indicator for whether a datapoint is OOD or not, and in many cases, the OOD data *consistently* gets higher likelihoods assigned (see the figure above).

On the flip side, the generative models that exhibit such pathological behavior are simultaneously capable of generating high-quality in-distribution data. Thus, the information required for OOD detection likely exists within these models, it just might not be the likelihood values alone. One important observation is that since OOD data is never generated from a generative model, the area around them (i.e., local probability mass around the OOD region) should have densities that integrate to a small probability. Thus using local probability *masses* instead of using mere probability *densities* seems like a more reasonable approach for OOD detection using generative models. To unravel this, we explore methods that take the entire high-dimensional density landscape induced by the generative model into consideration for OOD detection. 

This repository contains our ideas to tackle this problem alongside implementation of some recent relevant OOD detection baselines on a large set of models and datasets. The methods that are included are:

1. Likelihood ratio method for normalizing flows by [Ren et al.](https://arxiv.org/abs/1906.02845)
2. Complexity method for flow models by [Serrà et al.](https://arxiv.org/abs/1909.11480)
3. Likelihood ratio method for diffusion models by [Goodier et al.](https://arxiv.org/pdf/2310.17432.pdf) 
4. Reconstruction-based OOD detection with diffusions by [Graham et al.](https://arxiv.org/pdf/2211.07740.pdf).


## Setup

### Environment

**The codebase uses functional autodiff features that were recently added to `pytorch`. Make sure that your Python is `3.9` or higher; otherwise, some of the new autodiff functionalities we use might break. You should also install the `nflows` package from [here](https://github.com/HamidrezaKmK/nflows) which enables functional autodiff for the flow models such as the neural spline flows that we are using.**

Create the following conda environment to handle dependencies.

```bash
# This creates an environment called 'ood-detection':
conda env create -f env.yml 
# You can activate it as follows:
conda activate ood-detection
```

## Reproducibility Statement

We have an [old guide](./old.md) for reproducing our results. However, we have refactored the entire codebase and now included it in our [new repository](https://github.com/layer6ai-labs/dgm_geometry). We recommend using that version of our codebase in case you want to run our OOD detection approach.

