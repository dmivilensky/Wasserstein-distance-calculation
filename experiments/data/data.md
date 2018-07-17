# Experimental Data
The results of numerical experiments are stored here
## Experiments
1) Iterative Sinkhorn on random image data
    * 1_gammas -- list of gammas
    * 1_eps -- list of `1/log(epsilon)`
    * 1_ts -- list of `Sinkhorn(gamma[i, j], epsilon[i, j])` iterations with `1e-7` proximal epsilon
2) Sinkhorn on asymmetrical and symmetrical data
    * 2_gms -- list of gammas
    * 2\_ts\_sym -- list of `Sinkhorn(gamma[i, j], epsilon=1e-3, proxy_epsilon=1e-7)` iterations on symmetrical *C*
    * 2\_ts\_asym -- list of iterations on disbalanced *C* 
3) Sinkhorn on C-scaled data
    * 3_gammas -- list of gammas
    * 3\_coeffs -- list of *k* coefficients
    * 3\_ts -- list of `Sinkhorn(gamma[i, j], epsilon=1e-3, proxy_epsilon=1e-7)` iterations on *C\*k*
