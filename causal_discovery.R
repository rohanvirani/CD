library(pcalg)
library(CondIndTests)
trial <- subset(BH, select = c(age,rm,tax,ptratio,crim,zn))
suffStat <- list(C = cor(trial), n = nrow(trial))
skel.fit_kci <- skeleton(suffStat,indepTest = kci_skeleton,alpha=0.05,labels=colnames(trial))
if (require(Rgraphviz)) {
  ## show estimated Skeleton
  par(mfrow=c(1,2))
  plot(skel.fit_kci, main = "Estimated Skeleton using KCI")
}

skel.fit_gcm <- skeleton(suffStat,indepTest = gcm_skeleton,alpha=0.05,labels=colnames(trial))
if (require(Rgraphviz)) {
  ## show estimated Skeleton
  par(mfrow=c(1,2))
  plot(skel.fit_gcm, main = "Estimated Skeleton using GCM")
}

