kci_skeleton <- function(i,j,k,suffStat) {
  library(dHSIC)
  X <- trial[c(i)]
  Y <- trial[c(j)]
  if (length(k) == 0) {
    p <- dhsic.test(X,Y,alpha=0.05)$p.value
  }
  else {
    Z <- trial[c(k)]
    p <- KCI(X,Y,Z,alpha=0.05,gammaApprox=FALSE,GP=FALSE)$pvalue
  }
  return (p)
}

gcm_skeleton <- function(i,j,k,suffStat) {
  library(dHSIC)
  X <- data.matrix(trial[c(i)])
  Y <- data.matrix(trial[c(j)])
  if (length(k) == 0) {
    p <- dhsic.test(X,Y,alpha=0.05)$p.value
  }
  else {
    Z <- data.matrix(trial[c(k)])
    p <- gcm.test(X,Y,Z,alpha=0.05,regr.method="xgboost",plot.residuals = FALSE)$p.value
  }
  return (p)
}
