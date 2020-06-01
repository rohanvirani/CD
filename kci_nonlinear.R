library(CondIndTests)
library(pracma)
library(LaplacesDemon)
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
    rho_vals <- c(rho_vals,rho)
    for (i in 1:100) {
      sigma <- matrix(c(1,rho,rho,1),2,2)
      error <- rmvn(n,c(0,0),sigma)
      Z1 <- rnorm(n,2,1)
      X1 <- Z1*sin(15*Z1 + 0.0001) + error[,1]
      Y1 <- Z1*Z1*tanh(5*Z1 + + 0.004) + error[,2]
      if (rho == 0) {
        if (KCI(X1,Y1,Z1,alpha=0.05,gammaApprox=FALSE,GP=FALSE)$pvalue < 0.05) {
          t_1 = t_1 + 1
        }
      }
      if (rho != 0) {
        if (KCI(X1,Y1,Z1,alpha=0.05,gammaApprox=FALSE,GP=FALSE)$pvalue >= 0.05) {
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

df_1 <- data.frame(t1 = c(t1_complete))
write.csv(df_1,"~/Documents/M3R/CD/kci_nonlinear_t1.csv")

df_2 <- data.frame(t2 = c(t2_complete))
write.csv(df_2,"~/Documents/M3R/CD/kci_nonlinear_t2.csv")
