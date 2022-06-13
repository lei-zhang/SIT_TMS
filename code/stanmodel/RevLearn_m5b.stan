// Fictitious RL + instantaneous social information + SL (others’ action preference)
// [0 1]
data {
  int<lower=1> nSubjects;                              // number of subjects
  int<lower=1> nTrials;                                // number of trials 
  int<lower=1,upper=2> choice1_V[nSubjects,nTrials];   // 1st choices, 1 or 2
  int<lower=1,upper=2> choice1_R[nSubjects,nTrials]; 
  int<lower=1,upper=2> choice1_L[nSubjects,nTrials]; 
  int<lower=1,upper=2> choice2_V[nSubjects,nTrials];   // 2nd choices, 1 or 2
  int<lower=1,upper=2> choice2_R[nSubjects,nTrials];
  int<lower=1,upper=2> choice2_L[nSubjects,nTrials];
  int<lower=0,upper=1> chswtch_V[nSubjects,nTrials];   // choice switch, 0 or 1
  int<lower=0,upper=1> chswtch_R[nSubjects,nTrials];
  int<lower=0,upper=1> chswtch_L[nSubjects,nTrials];
  real<lower=-1,upper=1> reward_V[nSubjects,nTrials];  // outcome, 1 or -1, 'real' is faster than 'int'
  real<lower=-1,upper=1> reward_R[nSubjects,nTrials];
  real<lower=-1,upper=1> reward_L[nSubjects,nTrials];
  real<lower=0,upper=1>  wgtAgst_ms_V[nSubjects,nTrials]; // weighted against
  real<lower=0,upper=1>  wgtAgst_ms_R[nSubjects,nTrials];
  real<lower=0,upper=1>  wgtAgst_ms_L[nSubjects,nTrials];
  matrix<lower=0,upper=1>[nTrials,4] otherWith2_V[nSubjects];
  matrix<lower=0,upper=1>[nTrials,4] otherWith2_R[nSubjects];
  matrix<lower=0,upper=1>[nTrials,4] otherWith2_L[nSubjects];
  matrix<lower=0,upper=1>[nTrials,4] otherReward2_V[nSubjects];
  matrix<lower=0,upper=1>[nTrials,4] otherReward2_R[nSubjects];
  matrix<lower=0,upper=1>[nTrials,4] otherReward2_L[nSubjects];
}

transformed data {
  vector[2] initV;    // initial values for V
  int<lower=1> B;     // number of beta predictor
  matrix<lower=0,upper=1>[3,3] M; // design matrix for a within-subject effects coding
  matrix[nTrials,4]  wOthers[nSubjects];

  initV = rep_vector(0.0,2);    
  B = 5;
  M = [[1, 0, 0], [1, 1, 0], [1, 0, 1]];
  for (s in 1:nSubjects) {
    wOthers[s] = rep_matrix(0.25, nTrials,4);
  }
  /*  V  R  L    note that the order is always [V]ertex, [R]ight, [L]eft
    [ 1  0  0 ]
    [ 1  1  0 ]
    [ 1  0  1 ]
    parameters related to R/L-TPJ are coded as the difference w.r.t. the parameter of the Vertex condition
    e.g., b = [1 .2 .-1]';
    M * b --> [1, 1.2, 0.9]'
  */ 
}

parameters {
  // group-level parameters
  vector[3] lr_mu_pr;    // lr_mu before probit; [V, R-V, L-V] (diff w.r.t. V)
  vector<lower=0>[3] lr_sd; // standard deviation
  cholesky_factor_corr[3] lr_L; // Cholesky factor of correlation matrix, by convention called L 

  vector[3] beta_mu[B];
  vector<lower=0>[3] beta_sd[B];
  cholesky_factor_corr[3] beta_L[B];
  
  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  matrix[nSubjects,3] lr_raw;
  matrix[nSubjects,3] beta_raw[B];   // dim: [B, nSubjects, 3]
}

transformed parameters {
  // subject-level parameters
  matrix[nSubjects,3] lr_eff; // with effect coding, before transformation, [V, R-V, L-V]
  matrix<lower=0,upper=1>[nSubjects,3] lr; // actual, after transformation, [V, R, L] 
  matrix[nSubjects,3] beta_eff[B]; 
  matrix[nSubjects,3] beta[B];

  // Matt Trick (reparameterization) - multivariate
  lr_eff = rep_matrix(lr_mu_pr', nSubjects) + lr_raw * diag_pre_multiply(lr_sd, lr_L)';
  /* This is:
       [s_V, s_R, s_L] * L' * D' + [m_V, m_R, m_L]
       [..., ..., ...]             [..., ..., ...]
       [..., .. . ...]             [..., ..., ...]
     , where D is a diagonal matrix with the standard deviations on the diagonal;
     diag_pre_multiply(lr_sd, L) means diag_matrix(lr_sd) * L.

     ref: https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html
     also note: u*(diag(s) * L)' is equivalent to ((diag(s) * L)* u')'
  */

  for (j in 1:B) {
    beta_eff[j] = rep_matrix(beta_mu[j]', nSubjects) + beta_raw[j] * diag_pre_multiply(beta_sd[j], beta_L[j])';
  }

  // now convert to the actual parm to use
  lr = Phi_approx( lr_eff * M' );
  for (j in 1:B) {
    beta[j] = beta_eff[j] * M';
  }  
}

model {
  // hyperparameters
  lr_mu_pr ~ std_normal();
  lr_sd ~ normal(0,0.2);
  lr_L ~ lkj_corr_cholesky(2); // LKJ prior on (cholesky factor of) correlation matrix

  for (j in 1:B) {
    beta_mu[j] ~ std_normal(); 
    beta_sd[j] ~ normal(0,0.2);
    beta_L[j] ~ lkj_corr_cholesky(2);
  }
  
  // individual level
  to_vector(lr_raw) ~ std_normal();

  for (j in 1:B) {
    to_vector(beta_raw[j]) ~ std_normal();
  } 
 
  // subject loop and trial loop
  for (s in 1:nSubjects) {    
    // define the value and pe vectors
    vector[2] myValue_V; // values
    vector[2] myValue_R;
    vector[2] myValue_L;
    vector[2] otherValue_V;
    vector[2] otherValue_R;
    vector[2] otherValue_L;
    real pe_V; // prediction errors  
    real pe_R;
    real pe_L;
    real penc_V; // fictitious prediction errors  
    real penc_R;
    real penc_L;
    vector[2] valfun1_V;
    vector[2] valfun1_R;
    vector[2] valfun1_L;
    real valfun2_V;
    real valfun2_R;
    real valfun2_L;
    real valdiff_V;
    real valdiff_R;
    real valdiff_L;

    row_vector[4] otherReward_vec_V;
    row_vector[4] otherReward_vec_R;
    row_vector[4] otherReward_vec_L;

    myValue_V = initV;
    myValue_R = initV;
    myValue_L = initV;
    otherValue_V = initV;
    otherValue_R = initV;
    otherValue_L = initV;

    for (t in 1:nTrials) {
      // Vertex
      valfun1_V = beta[1,s,1] * myValue_V + beta[2,s,1] * otherValue_V;
      choice1_V[s,t] ~ categorical_logit( valfun1_V );
      valdiff_V = valfun1_V[choice1_V[s,t]] - valfun1_V[3 - choice1_V[s,t]];

      valfun2_V = beta[3,s,1] + beta[4,s,1] * valdiff_V + beta[5,s,1] * wgtAgst_ms_V[s,t];
      chswtch_V[s,t] ~ bernoulli_logit(valfun2_V);

      pe_V   =  reward_V[s,t] - myValue_V[choice2_V[s,t]]; // prediction error
      penc_V = -reward_V[s,t] - myValue_V[3 - choice2_V[s,t]]; 
      myValue_V[choice2_V[s,t]] += lr[s,1] * pe_V; // value updating
      myValue_V[3 - choice2_V[s,t]] += lr[s,1] * penc_V;

      otherReward_vec_V = wOthers[s,t] .* otherReward2_V[s,t];
      otherValue_V[choice2_V[s,t]]   = sum( otherWith2_V[s,t] .* otherReward_vec_V );
      otherValue_V[3-choice2_V[s,t]] = sum( (1-otherWith2_V[s,t]) .* otherReward_vec_V );

      // R-TPJ
      valfun1_R = beta[1,s,2] * myValue_R + beta[2,s,2] * otherValue_R;
      choice1_R[s,t] ~ categorical_logit( valfun1_R );
      valdiff_R = valfun1_R[choice1_R[s,t]] - valfun1_R[3 - choice1_R[s,t]];

      valfun2_R = beta[3,s,2] + beta[4,s,2] * valdiff_R + beta[5,s,2] * wgtAgst_ms_R[s,t];
      chswtch_R[s,t] ~ bernoulli_logit(valfun2_R);

      pe_R   =  reward_R[s,t] - myValue_R[choice2_R[s,t]];
      penc_R = -reward_R[s,t] - myValue_R[3 - choice2_R[s,t]]; 
      myValue_R[choice2_R[s,t]] += lr[s,2] * pe_R;
      myValue_R[3 - choice2_R[s,t]] += lr[s,2] * penc_R;

      otherReward_vec_R = wOthers[s,t] .* otherReward2_R[s,t];
      otherValue_R[choice2_R[s,t]]   = sum( otherWith2_R[s,t] .* otherReward_vec_R );
      otherValue_R[3-choice2_R[s,t]] = sum( (1-otherWith2_R[s,t]) .* otherReward_vec_R );

      // L-TPJ
      valfun1_L = beta[1,s,3] * myValue_L + beta[2,s,3] * otherValue_L;
      choice1_L[s,t] ~ categorical_logit( valfun1_L );
      valdiff_L = valfun1_L[choice1_L[s,t]] - valfun1_L[3 - choice1_L[s,t]];

      valfun2_L = beta[3,s,3] + beta[4,s,3] * valdiff_L + beta[5,s,3] * wgtAgst_ms_L[s,t];
      chswtch_L[s,t] ~ bernoulli_logit(valfun2_L);

      pe_L   =  reward_L[s,t] - myValue_L[choice2_L[s,t]];
      penc_L = -reward_L[s,t] - myValue_L[3 - choice2_L[s,t]]; 
      myValue_L[choice2_L[s,t]] += lr[s,3] * pe_L;
      myValue_L[3 - choice2_L[s,t]] += lr[s,3] * penc_L;

      otherReward_vec_L = wOthers[s,t] .* otherReward2_L[s,t];
      otherValue_L[choice2_L[s,t]]   = sum( otherWith2_L[s,t] .* otherReward_vec_L );
      otherValue_L[3-choice2_L[s,t]] = sum( (1-otherWith2_L[s,t]) .* otherReward_vec_L );
    } // trial
  } // subj
}

generated quantities {
  vector<lower=-1,upper=1>[3] lr_mu;
  matrix[3,3] lr_corr; // correlation coefficient
  matrix[3,3] beta_corr[B];
  
  real log_likc1_V[nSubjects];
  real log_likc1_R[nSubjects];
  real log_likc1_L[nSubjects];
  real log_likc2_V[nSubjects];
  real log_likc2_R[nSubjects];
  real log_likc2_L[nSubjects];

  // recover the parameters
  lr_mu = inverse(M) * Phi_approx(M * lr_mu_pr); // effect coding --> difference
  lr_corr = multiply_lower_tri_self_transpose(lr_L);
  
  for (j in 1:B) {
    beta_corr[j] = multiply_lower_tri_self_transpose(beta_L[j]);
  }

  { // compute the log-likelihood 
    for (s in 1:nSubjects) {    
      // define the value and pe vectors
      vector[2] myValue_V; // values
      vector[2] myValue_R;
      vector[2] myValue_L;
      vector[2] otherValue_V;
      vector[2] otherValue_R;
      vector[2] otherValue_L;
      real pe_V; // prediction errors  
      real pe_R;
      real pe_L;
      real penc_V; // fictitious prediction errors  
      real penc_R;
      real penc_L;
      vector[2] valfun1_V;
      vector[2] valfun1_R;
      vector[2] valfun1_L;
      real valfun2_V;
      real valfun2_R;
      real valfun2_L;
      real valdiff_V;
      real valdiff_R;
      real valdiff_L;

      row_vector[4] otherReward_vec_V;
      row_vector[4] otherReward_vec_R;
      row_vector[4] otherReward_vec_L;

      myValue_V = initV;
      myValue_R = initV;
      myValue_L = initV;
      otherValue_V = initV;
      otherValue_R = initV;
      otherValue_L = initV;

      log_likc1_V[s] = 0;
      log_likc1_R[s] = 0;
      log_likc1_L[s] = 0;
      log_likc2_V[s] = 0;
      log_likc2_R[s] = 0;
      log_likc2_L[s] = 0;

      for (t in 1:nTrials) {
        // Vertex
        valfun1_V = beta[1,s,1] * myValue_V + beta[2,s,1] * otherValue_V;
        log_likc1_V[s] += categorical_logit_lpmf(choice1_V[s,t] | valfun1_V);
        valdiff_V = valfun1_V[choice1_V[s,t]] - valfun1_V[3 - choice1_V[s,t]];

        valfun2_V = beta[3,s,1] + beta[4,s,1] * valdiff_V + beta[5,s,1] * wgtAgst_ms_V[s,t];
        log_likc2_V[s] += bernoulli_logit_lpmf(chswtch_V[s,t] | valfun2_V);

        pe_V   =  reward_V[s,t] - myValue_V[choice2_V[s,t]]; // prediction error
        penc_V = -reward_V[s,t] - myValue_V[3 - choice2_V[s,t]]; 
        myValue_V[choice2_V[s,t]] += lr[s,1] * pe_V; // value updating
        myValue_V[3 - choice2_V[s,t]] += lr[s,1] * penc_V;

        otherReward_vec_V = wOthers[s,t] .* otherReward2_V[s,t];
        otherValue_V[choice2_V[s,t]]   = sum( otherWith2_V[s,t] .* otherReward_vec_V );
        otherValue_V[3-choice2_V[s,t]] = sum( (1-otherWith2_V[s,t]) .* otherReward_vec_V );
      
        // R-TPJ
        valfun1_R = beta[1,s,2] * myValue_R + beta[2,s,2] * otherValue_R;
        log_likc1_R[s] += categorical_logit_lpmf(choice1_R[s,t] | valfun1_R);
        valdiff_R = valfun1_R[choice1_R[s,t]] - valfun1_R[3 - choice1_R[s,t]];

        valfun2_R = beta[3,s,2] + beta[4,s,2] * valdiff_R + beta[5,s,2] * wgtAgst_ms_R[s,t];
        log_likc2_R[s] += bernoulli_logit_lpmf(chswtch_R[s,t] | valfun2_R);

        pe_R   =  reward_R[s,t] - myValue_R[choice2_R[s,t]];
        penc_R = -reward_R[s,t] - myValue_R[3 - choice2_R[s,t]]; 
        myValue_R[choice2_R[s,t]] += lr[s,2] * pe_R;
        myValue_R[3 - choice2_R[s,t]] += lr[s,2] * penc_R;

        otherReward_vec_R = wOthers[s,t] .* otherReward2_R[s,t];
        otherValue_R[choice2_R[s,t]]   = sum( otherWith2_R[s,t] .* otherReward_vec_R );
        otherValue_R[3-choice2_R[s,t]] = sum( (1-otherWith2_R[s,t]) .* otherReward_vec_R );
      
        // L-TPJ
        valfun1_L = beta[1,s,3] * myValue_L + beta[2,s,3] * otherValue_L;
        log_likc1_L[s] += categorical_logit_lpmf(choice1_L[s,t] | valfun1_L);
        valdiff_L = valfun1_L[choice1_L[s,t]] - valfun1_L[3 - choice1_L[s,t]];

        valfun2_L = beta[3,s,3] + beta[4,s,3] * valdiff_L + beta[5,s,3] * wgtAgst_ms_L[s,t];
        log_likc2_L[s] += bernoulli_logit_lpmf(chswtch_L[s,t] | valfun2_L);

        pe_L   =  reward_L[s,t] - myValue_L[choice2_L[s,t]];
        penc_L = -reward_L[s,t] - myValue_L[3 - choice2_L[s,t]]; 
        myValue_L[choice2_L[s,t]] += lr[s,3] * pe_L;
        myValue_L[3 - choice2_L[s,t]] += lr[s,3] * penc_L;

        otherReward_vec_L = wOthers[s,t] .* otherReward2_L[s,t];
        otherValue_L[choice2_L[s,t]]   = sum( otherWith2_L[s,t] .* otherReward_vec_L );
        otherValue_L[3-choice2_L[s,t]] = sum( (1-otherWith2_L[s,t]) .* otherReward_vec_L );
      
      } // trial
    } // subj
  } // local

}
