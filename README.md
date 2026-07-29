# jumufraktiv
A package that returns model evidence, posterior density, posterior predictive, posterior MGF and posterior moments by calculating fractional derivatives of prior MGF

## Things to check before start
1. This package only supports strictly positive parameters. Does not matter if the parameter space takes the entire positive real line or not.
2. The options of likelihoods that this package supports is limited to Poisson, Laplace with known mean, normal with known mean, Rayleigh, Maxwell-Boltzmann, gamma with known shape, inverse gamma with known shape, Levy with known mean, Weibull with known rho=shape/scale, Burr XII with known c, Pareto with known scale, Dagum with known a and b, and Gompertz with known scale.
3. There is no limitations on the prior distributions this package supports, except for improper priors with MGF values of ∞ on the negative half of the real line.
### If any of the above is found not suitable for your particular use, then please consider other packages available.
