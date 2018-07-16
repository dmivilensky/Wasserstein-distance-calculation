# Experimental Data
The results of numerical experiments are stored here
## Experiments
1) Iterative Sinkhorn on random image data
    * 1_gammas -- list of gammas
    * 1_eps -- list of epsilons
    * 1_ts -- list of `Sinkhorn(gamma[i, j], epsilon[i, j])` iterations with `1e-7` proximal epsilon
2) Sinkhorn on asymmetrical and symmetrical data
    * 2_gms -- list of gammas
    * 2\_ts\_sym -- list of `Sinkhorn(gamma[i, j], epsilon=1e-3, proxy_0epsilon=1e-7)` iterations on symmetrical *C*
    * 2\_ts\_asym -- list of iterations on disbalanced *C* 
