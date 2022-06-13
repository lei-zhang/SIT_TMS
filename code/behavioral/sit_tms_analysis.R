library(R.matlab)
library(reshape2)
library(lme4)
library(bruceR)
###
library(lmerTest)
detach('package:lmerTest',unload = T)


####################################################################################################
#### test the main findings - choice1 switch probability, within trial
####################################################################################################
#### load data ----------------
ns = 31
data_dir <- 'data/preprocessed/'
pSwi <- readMat(paste0(data_dir, 'lmer_pSwi_sub_mat_prep.mat'))
pSwi <- as.data.frame(pSwi$pSwi.sub.mat)
dim(pSwi) # [93 7]
str(pSwi)

#### prepare data ----------------
pSwi$V6 <- as.factor(pSwi$V6)
pSwi$V7 <- as.factor(pSwi$V7)
levels(pSwi$V6) <- c('vertex', 'rtpj', 'ltpj')
levels(pSwi$V7) <- c('male', 'female')
pSwi$sid = 1:ns

pSwi <- pSwi[c(8,7,6,1,2,3,4,5)]
colnames(pSwi) <- c('sid','sex','stim','with31','with40','agst22','agst31','agst40')
pSwi$coh22 = pSwi$agst22 - pSwi$agst22
pSwi$coh31 = pSwi$agst31 - pSwi$with31
pSwi$coh40 = pSwi$agst40 - pSwi$with40

pSwi <- pSwi[c(1:3, 9:11)]
X1 <- reshape2::melt(pSwi, id.vars = c('sid','sex','stim'),
           measure.vars = c('coh22','coh31','coh40'))
X1$coh <- rep(c(2,3,4), each = ns * 3)
X1$coh <- as.factor(X1$coh)
levels(X1$coh) <- c('2:2','3:1','4:0')
colnames(X1)[4:6] <- c('cond', 'pSwi', 'coh')

aggregate(pSwi ~ coh * stim, data=X1, mean)

#### stats ----------------
fx <- lmer(pSwi ~ stim * coh + (1+coh+stim|sid), data=X1)
anova(fx)
bruceR::HLM_summary(fx)
lsmeans::lsmeans(fx, list(pairwise ~ stim * coh), adjust = "tukey")

detach('package:lmerTest',unload = T)


####################################################################################################
#### test the main findings - RT, within trial
####################################################################################################

#### load data ----------------
ns = 31
data_dir <- 'data/preprocessed/'
rt <- readMat(paste0(data_dir, 'lmer_RT_sub_mat_prep.mat'))
rt <- as.data.frame(rt$RT.sub.mat)
dim(rt) # [93 7]
str(rt)

#### prepare data ----------------
rt$V6 <- as.factor(rt$V6)
rt$V7 <- as.factor(rt$V7)
levels(rt$V6) <- c('vertex', 'rtpj', 'ltpj')
levels(rt$V7) <- c('male', 'female')
rt$sid = 1:ns

rt <- rt[c(8,7,6,1,2,3,4,5)]
colnames(rt) <- c('sid','sex','stim','with31','with40','agst22','agst31','agst40')
rt$coh22 = rt$agst22 - rt$agst22
rt$coh31 = rt$agst31 - rt$with31
rt$coh40 = rt$agst40 - rt$with40

rt <- rt[c(1:3, 9:11)]
X2 <- reshape2::melt(rt, id.vars = c('sid','sex','stim'),
                     measure.vars = c('coh22','coh31','coh40'))
X2$coh <- rep(c(2,3,4), each = ns * 3)
X2$coh <- as.factor(X2$coh)
levels(X2$coh) <- c('2:2','3:1','4:0')
colnames(X2)[4:6] <- c('cond', 'rt', 'coh')

aggregate(rt ~ coh * stim , data=X2, mean)

#### stats ----------------
library(lmerTest)
fx <- lmer(rt ~ stim * coh + (1+coh+stim|sid), data=X2)
bruceR::HLM_summary(fx)
lsmeans::lsmeans(fx, list(pairwise ~ stim * coh), adjust = "tukey")

detach('package:lmerTest',unload = T)



####################################################################################################
#### test the main findings - within-trial pCorrect of C2 on trial t, only Switch trials
####################################################################################################
#### load data ----------------
ns = 31
data_dir <- 'data/preprocessed/'
pCorr_WT <- readMat(paste0(data_dir, 'lmer_pCorr_WT_sub_mat_prep.mat'))
pCorr_WT <- as.data.frame(pCorr_WT$pCorr.WT.sub.mat)
dim(pCorr_WT) # [186 8]
str(pCorr_WT)

#### prepare data ----------------
pCorr_WT$V6 <- as.factor(pCorr_WT$V6)
pCorr_WT$V7 <- as.factor(pCorr_WT$V7)
pCorr_WT$V8 <- as.factor(pCorr_WT$V8)
levels(pCorr_WT$V6) <- c('stay', 'switch')
levels(pCorr_WT$V7) <- c('vertex', 'rtpj', 'ltpj')
levels(pCorr_WT$V8) <- c('male', 'female')
pCorr_WT$sid = 1:ns

pCorr_WT <- pCorr_WT[c(9,8,7,6,1,2,3,4,5)]
colnames(pCorr_WT) <- c('sid','sex','stim','StSw','with31','with40','agst22','agst31','agst40')
pCorr_WT= pCorr_WT[pCorr_WT$StSw == 'switch',]
pCorr_WT = pCorr_WT[, c(1:3, 5:9)]

pCorr_WT$coh22 = pCorr_WT$agst22 - pCorr_WT$agst22
pCorr_WT$coh31 = pCorr_WT$agst31 - pCorr_WT$with31
pCorr_WT$coh40 = pCorr_WT$agst40 - pCorr_WT$with40

pCorr_WT <- pCorr_WT[c(1:3, 9:11)]
X3 <- reshape2::melt(pCorr_WT, id.vars = c('sid','sex','stim'),
                     measure.vars = c('coh22','coh31','coh40'))
X3$coh <- rep(c(2,3,4), each = ns * 3)
X3$coh <- as.factor(X3$coh)
levels(X3$coh) <- c('2:2','3:1','4:0')
colnames(X3)[4:6] <- c('cond', 'pCorr_WT', 'coh')

aggregate(pCorr_WT ~ coh * stim, data=X3, mean)

#### stats ----------------

library(lmerTest)
fx <- lmer(pCorr_WT ~ stim * coh + (1+coh+stim|sid), data=X3)
bruceR::HLM_summary(fx)
lsmeans::lsmeans(fx, list(pairwise ~ stim * coh), adjust = "tukey")

detach('package:lmerTest',unload = T)

