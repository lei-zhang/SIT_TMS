sitTMS_run <- function(modelStr, test = TRUE, vb = FALSE, saveFit = FALSE, suffix=NULL,
                       nSubj = NULL, site = NULL, seed = NULL,
                       nIter = 2000, nwarmup = NULL, nCore = NULL,
                       adapt = 0.9, treedepth = 10, nThinn = 1 ) {
    
    # estimating only the choice, but not the bet / confidence
    
    library(rstan); library(loo); library(hBayesDM)
    L <- list()
    
    #### load data #### ===========================================================================
    load("data/preprocessed/dataList_rvl_n31.rdata")
    
    #### preparation for running stan #### ============================================================
    # model string in a separate .stan file
    modelFile <- paste0("code/stanmodels/",modelStr,".stan")
    
    # setup up Stan configuration
    if (test == TRUE) {
        options(mc.cores = 1)
        nSamples <- 4
        nChains  <- 1 
        nBurnin  <- 0
        nThin    <- 1
        est_seed = sample.int(.Machine$integer.max, 1)
        
    } else {
        if (is.null(nCore)) {
            options(mc.cores = 4) 
        } else {
            options(mc.cores = nCore)  
        }
        
        if (is.null(seed)) {
            est_seed = sample.int(.Machine$integer.max, 1) 
        } else {
            est_seed = seed  
        }
        
        nSamples <- nIter
        nChains  <- 4
        if (is.null(nwarmup)) {
            nBurnin <- floor(nSamples/2)
        } else {
            nBurnin = nwarmup
        }
        nThin    <- nThinn
    }
    
    # parameter of interest (this could save both memory and space)
    poi <- create_pois(modelStr)
    
    #### run stan ####  ==============================================================================
    startTime = Sys.time(); print(startTime)
    cat("\nDetails:\n")
    cat("Estimating:", modelStr, "... \n")
    cat("Variant: ", modelStr, suffix, " ... \n", sep = "")
    cat("Running with", dataList$nSubjects, "participants... \n")
    cat("\nMethods:\n")
    if (vb == TRUE) {
        cat(" # Calling Variational Bayes methods in Stan... \n")
    } else {
        cat(" # Calling", nChains, "MCMC chains in Stan... \n")
        cat(" # of MCMC samples (per chain) = ", nSamples, "\n")
        cat(" # of burn-in samples          = ", nBurnin, "\n")
    }
    
    rstan_options(auto_write = TRUE)
    stanmodel <- rstan::stan_model(modelFile)
    
    if (vb == TRUE) {
        stanfit <- rstan::vb(
            object  = stanmodel,
            data    = dataList,
            pars    = poi$pars,
            init    = "random")
        
    } else {
        stanfit <- rstan::sampling(
            object  = stanmodel,
            data    = dataList,
            pars    = poi$pars,
            chains  = nChains,
            iter    = nSamples,
            warmup  = nBurnin,
            thin    = nThin,
            init    = "random",
            seed    = est_seed,
            control = list(adapt_delta = adapt, max_treedepth = treedepth),
            verbose = FALSE)
    } 
    
    cat("Finishing", modelStr, "model estimation ... \n")
    endTime = Sys.time(); print(endTime)  
    cat("It took",as.character.Date(endTime - startTime), "\n")
    
    L$data <- dataList
    L$fit  <- stanfit
    
    class(L) <- "hBayesDM"
    
    if (saveFit == TRUE) {
        if (vb == TRUE) {
            saveRDS(L, file = paste0('_stanfits/', modelStr, suffix, '_VB.RData'))
        } else {
            saveRDS(L, file = paste0('_stanfits/', modelStr, suffix, '.RData'))
        }
    }
    
    if (!vb) {
        # model diag information
        rh = brms::rhat(L$fit, pars = poi$pars_chk)
        cat(' # --- rhat range:', round(range(rh, na.rm = TRUE), 4), '\n')
        cat(' # --- rhat above 1.10: N =', sum(rh > 1.1, na.rm = TRUE), '\n')
        
        cat(' # --- LOOIC: \n')
        print(extract_looic_C(L,nCore=4, nChains, nSamples, nBurnin)$looicALL)
    }
    
    return(L)
}  # function run_model()


#### nested functions #### ===========================================================================
create_pois <- function(model){
    pois <- list()
    
    if (model == "RevLearn_m1a" || model == "RevLearn_m1b" ||
        model == "RevLearn_m2b" || model == "RevLearn_m4b" ||
        model == "RevLearn_m4b_v2" || model == "RevLearn_m4b_v3" ||
        model == "RevLearn_m5b" || model == "RevLearn_m5b_v2") {
        pars = c("lr_mu", "beta_mu", 
                 "lr_sd", "beta_sd", 
                 "lr_corr", "beta_corr",
                 "lr", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr_mu", "beta_mu", 
                     "lr_sd", "beta_sd", 
                     "lr_corr", "beta_corr",
                     "lr", "beta")
        
    } else if ( model == "RevLearn_m1c" || 
                model == "RevLearn_m2") {
        pars = c("lr_pos_mu", "lr_neg_mu", "beta_mu", 
                 "lr_pos_sd", "lr_neg_sd", "beta_sd", 
                 "lr_pos_corr", "lr_neg_corr", "beta_corr",
                 "lr_pos", "lr_neg", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr_pos_mu", "lr_neg_mu", "beta_mu", 
                     "lr_pos_sd", "lr_neg_sd", "beta_sd", 
                     "lr_pos_corr", "lr_neg_corr", "beta_corr",
                     "lr_pos", "lr_neg", "beta")
        
    } else if ( model == "RevLearn_m1d") {
        pars = c("lr0_mu", "eta_mu", "k_mu", "beta_mu", 
                 "lr0_sd", "eta_sd", "k_sd", "beta_sd", 
                 "lr0_corr", "eta_corr", "k_corr", "beta_corr",
                 "lr0", "eta", "k", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr0_mu", "eta_mu", "k_mu", "beta_mu", 
                     "lr0_sd", "eta_sd", "k_sd", "beta_sd", 
                     "lr0_corr", "eta_corr", "k_corr", "beta_corr",
                     "lr0", "eta", "k", "beta")
        
    } else if ( model == "RevLearn_m3b" || model == "RevLearn_m3b_v3") {
        pars = c("lr_mu", "lr_oth_mu", "tau_oth_mu", "beta_mu", 
                 "lr_sd", "lr_oth_sd", "tau_oth_sd", "beta_sd", 
                 "lr_corr", "lr_oth_corr", "tau_oth_corr", "beta_corr",
                 "lr", "lr_oth", "tau_oth", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr_mu", "lr_oth_mu", "tau_oth_mu", "beta_mu", 
                     "lr_sd", "lr_oth_sd", "tau_oth_sd", "beta_sd", 
                     "lr_corr", "lr_oth_corr", "tau_oth_corr", "beta_corr",
                     "lr", "lr_oth", "tau_oth", "beta")
            
    } else if ( model == "RevLearn_m3b_v2") {
        pars = c("lr_mu", "lr_oth_mu", "beta_mu", 
                 "lr_sd", "lr_oth_sd", "beta_sd", 
                 "lr_corr", "lr_oth_corr", "beta_corr",
                 "lr", "lr_oth", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr_mu", "lr_oth_mu", "beta_mu", 
                     "lr_sd", "lr_oth_sd", "beta_sd", 
                     "lr_corr", "lr_oth_corr", "beta_corr",
                     "lr", "lr_oth", "beta")
        
    } else if ( model == "RevLearn_m6b" || model == "RevLearn_m6b_v2") {
        pars = c("lr_mu", "disc_mu", "beta_mu", 
                 "lr_sd", "disc_sd", "beta_sd", 
                 "lr_corr", "disc_corr", "beta_corr",
                 "lr", "disc", "beta", 
                 "log_likc1_V", "log_likc1_R", "log_likc1_L", 
                 "log_likc2_V", "log_likc2_R", "log_likc2_L", 
                 "lp__")
        pars_chk = c("lr_mu", "disc_mu", "beta_mu", 
                     "lr_sd", "disc_sd", "beta_sd", 
                     "lr_corr", "disc_corr", "beta_corr",
                     "lr", "disc", "beta")
        
    } 
    
    pois$pars = pars
    pois$pars_chk = pars_chk
    return(pois)
} # function

#### end of function ####
