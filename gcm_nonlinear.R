library(GeneralisedCovarianceMeasure)
library(pracma)
t1_complete <- vector()
t2_complete <- vector()
t1_all_vals <- vector()
t2_all_vals <- vector()
rho_vals <- vector()
for (n in c(50,100,150,200))  { 
  t1_all_vals <- vector()
  t2_all_vals <- vector()
  for (rho in linspace(0,0.95,20)) {
    t_1 = 0
    t_2 = 0
    for (i in 1:100) {
      simga <- matrix(c(1,rho,rho,1),2,2)
      e_X <- rmvn(n,c(0,0),sigma[,1])
      simga_y <- matrix(c(1,0,0,1),2,2)
      e_Y <- rmvn(n,c(0,0),sigma[,2])
      Z1 <- rnorm(n,0,1)
      X1 <- 15*Z1 + 0.0001 + e_X
      Y1 <- 5*Z1 + + 0.004 + e_Y
      if (rho == 0) {
        if (gcm.test(X1,Y1,Z1,alpha=0.05,regr.method="gam",plot.residuals = TRUE)$p.value < 0.05) {
          t_1 = t_1 + 1
        }
      }
      if (rho != 0) {
        if (gcm.test(X1,Y1,Z1,alpha=0.05,regr.method="gam",plot.residuals = TRUE)$p.value >= 0.05) {
          t_2 = t_2 + 1
        }
      }
    }
    if (rho == 0) {
      t1_all_vals <- c(t1_all_vals,t_1)
    }
    if (rho != 0) {
      t2_all_vals <- c(t2_all_vals,t_2)
    }
  }
  t1_complete <- c(t1_complete,t1_all_vals)
  t2_complete <- c(t2_complete,t2_all_vals)
}

