library(GeneralisedCovarianceMeasure)
library(CondIndTests)
p_1 = 0
for (i in 1:100) {
  Z1 <- rnorm(100,0,1)
  X1 <- 15*Z1 + 0.0001 
  Y1 <- 5*Z1
  if (gcm.test(X1,Y1,Z1,alpha=0.05,regr.method="gam",plot.residuals = TRUE)$reject == TRUE) {
    p_1= p_1 + 1
  }
}
p_2 = 0
for (i in 1:100) {
  Z2 <- rnorm(100,0,1)
  X2 <- Z2*cos(sin(8*Z2*Z2*Z2)) + rnorm(100,0,1)  
  Y2 <- Z2*Z2*tanh(Z2*Z2 + 7*Z*sin(2-Z)) + rnorm(100,0,1)
  if (gcm.test(X2,Y2,Z2,alpha=0.05,regr.method="gam",plot.residuals = TRUE)$reject == TRUE) {
    p_2= p_2 + 1
  }
}

