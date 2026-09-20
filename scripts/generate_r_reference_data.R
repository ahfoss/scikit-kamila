# Generate reference test datasets and ground-truth KAMILA results from R package 'kamila'
# for validation in scikit-kamila.

library(kamila)
library(jsonlite)

set.seed(12345)

out_dir <- "kamila/tests/data"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# -----------------------------------------------------------------------------
# Helper: Run kamilaLoopCpp with explicit initialization
# -----------------------------------------------------------------------------
run_deterministic_kamila <- function(con_mat, cat_mat, num_clust, init_means, init_log_probs,
                                     num_lev, con_weights, cat_weights, cat_bw, max_iter,
                                     has_con = TRUE, has_cat = TRUE) {
  res <- kamila:::kamilaLoopCpp(
    conVarMat_ = if (has_con) con_mat else NULL,
    catFactorNum_ = if (has_cat) cat_mat else NULL,
    conWeights = con_weights,
    catWeights = cat_weights,
    initMeans_ = if (has_con) init_means else NULL,
    initLogProbs_ = if (has_cat) init_log_probs else list(),
    numLev = if (has_cat) num_lev else integer(0),
    catBw = cat_bw,
    numClust = as.integer(num_clust),
    maxIter = as.integer(max_iter),
    verbose = TRUE,
    hasCon = has_con,
    hasCat = has_cat
  )
  return(res)
}

# -----------------------------------------------------------------------------
# 1. Dataset 1: Small Mixed Data (N=12, P_con=2, P_cat=2, K=2)
# -----------------------------------------------------------------------------
cat("Generating Dataset 1 (Small Mixed)...\n")
n1 <- 12
con_mat_small <- matrix(c(
  1.0, 2.0,
  1.2, 2.1,
  0.8, 1.9,
  1.1, 2.3,
  1.3, 1.8,
  0.9, 2.2,
  5.0, 8.0,
  5.2, 8.2,
  4.8, 7.8,
  5.1, 8.3,
  5.3, 7.9,
  4.9, 8.1
), nrow = n1, ncol = 2, byrow = TRUE)

cat_mat_small <- matrix(as.integer(c(
  1, 1,
  1, 2,
  1, 1,
  2, 1,
  1, 2,
  1, 1,
  2, 3,
  2, 3,
  3, 2,
  2, 3,
  3, 3,
  2, 2
)), nrow = n1, ncol = 2, byrow = TRUE)

num_lev_small <- as.integer(c(3, 3)) # 3 levels each
con_weights_small <- c(1.0, 1.0)
cat_weights_small <- c(1.0, 1.0)
cat_bw_small <- 0.025
k_small <- 2L

# Explicit initial means:
init_means_small <- matrix(c(
  1.0, 2.0,
  5.0, 8.0
), nrow = k_small, ncol = 2, byrow = TRUE)

# Explicit initial log probabilities (K=2 x 3 levels):
# Variable 1:
init_lp_small_v1 <- matrix(log(c(
  0.7, 0.2, 0.1,
  0.1, 0.6, 0.3
)), nrow = k_small, ncol = 3, byrow = TRUE)
# Variable 2:
init_lp_small_v2 <- matrix(log(c(
  0.6, 0.3, 0.1,
  0.1, 0.3, 0.6
)), nrow = k_small, ncol = 3, byrow = TRUE)
init_log_probs_small <- list(init_lp_small_v1, init_lp_small_v2)

# Run 1 iteration:
res_small_1iter <- run_deterministic_kamila(
  con_mat = con_mat_small,
  cat_mat = cat_mat_small,
  num_clust = k_small,
  init_means = init_means_small,
  init_log_probs = init_log_probs_small,
  num_lev = num_lev_small,
  con_weights = con_weights_small,
  cat_weights = cat_weights_small,
  cat_bw = cat_bw_small,
  max_iter = 1L
)

# Run converged (max_iter = 20):
res_small_conv <- run_deterministic_kamila(
  con_mat = con_mat_small,
  cat_mat = cat_mat_small,
  num_clust = k_small,
  init_means = init_means_small,
  init_log_probs = init_log_probs_small,
  num_lev = num_lev_small,
  con_weights = con_weights_small,
  cat_weights = cat_weights_small,
  cat_bw = cat_bw_small,
  max_iter = 20L
)

data_small <- list(
  description = "Small mixed dataset with 12 observations, 2 continuous vars, 2 categorical vars (3 levels each), K=2",
  n_samples = n1,
  n_con = 2,
  n_cat = 2,
  num_clust = k_small,
  num_lev = num_lev_small,
  con_weights = con_weights_small,
  cat_weights = cat_weights_small,
  cat_bw = cat_bw_small,
  con_data = con_mat_small,
  cat_data = cat_mat_small - 1L, # 0-indexed for Python
  init_means = init_means_small,
  init_log_probs = init_log_probs_small,
  one_iteration = list(
    final_membership = as.numeric(res_small_1iter$finalMemb) - 1L, # 0-indexed
    final_means = res_small_1iter$finalMeans,
    final_log_probs = res_small_1iter$finalLogProbs,
    total_log_lik = res_small_1iter$totalLogLik,
    cat_log_lik = res_small_1iter$catLogLik,
    win_dist = res_small_1iter$winDist,
    num_iter = res_small_1iter$numIter
  ),
  converged = list(
    final_membership = as.numeric(res_small_conv$finalMemb) - 1L, # 0-indexed
    final_means = res_small_conv$finalMeans,
    final_log_probs = res_small_conv$finalLogProbs,
    total_log_lik = res_small_conv$totalLogLik,
    cat_log_lik = res_small_conv$catLogLik,
    win_dist = res_small_conv$winDist,
    num_iter = res_small_conv$numIter
  )
)

write_json(data_small, file.path(out_dir, "reference_small_mixed.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10)

# -----------------------------------------------------------------------------
# 2. Dataset 2: Medium Mixed Data (N=60, P_con=3, P_cat=2, K=3)
# -----------------------------------------------------------------------------
cat("Generating Dataset 2 (Medium Mixed)...\n")
set.seed(42)
n2 <- 60
k2 <- 3L
con_mat_med <- rbind(
  matrix(rnorm(20 * 3, mean = 1.0, sd = 0.5), nrow = 20, ncol = 3),
  matrix(rnorm(20 * 3, mean = 6.0, sd = 0.6), nrow = 20, ncol = 3),
  matrix(rnorm(20 * 3, mean = 11.0, sd = 0.5), nrow = 20, ncol = 3)
)

cat_mat_med <- rbind(
  cbind(sample(1:2, 20, replace = TRUE, prob = c(0.8, 0.2)), sample(1:3, 20, replace = TRUE, prob = c(0.7, 0.2, 0.1))),
  cbind(sample(1:2, 20, replace = TRUE, prob = c(0.2, 0.8)), sample(1:3, 20, replace = TRUE, prob = c(0.1, 0.7, 0.2))),
  cbind(sample(1:2, 20, replace = TRUE, prob = c(0.5, 0.5)), sample(1:3, 20, replace = TRUE, prob = c(0.2, 0.2, 0.6)))
)

num_lev_med <- as.integer(c(2, 3))
con_weights_med <- c(1.0, 1.0, 1.0)
cat_weights_med <- c(1.0, 1.0)
cat_bw_med <- 0.025

# Fixed initial cluster assignment vector (alternating / random partition)
set.seed(999)
init_memb_med <- sample(1:k2, n2, replace = TRUE)

# Initial means and log-probs computed deterministically from init_memb_med:
init_means_med <- matrix(0.0, nrow = k2, ncol = 3)
for (k in 1:k2) {
  idx <- which(init_memb_med == k)
  init_means_med[k, ] <- colMeans(con_mat_med[idx, , drop = FALSE])
}

init_log_probs_med <- list()
for (j in 1:ncol(cat_mat_med)) {
  prob_mat <- matrix(0.0, nrow = k2, ncol = num_lev_med[j])
  for (k in 1:k2) {
    idx <- which(init_memb_med == k)
    counts <- table(factor(cat_mat_med[idx, j], levels = 1:num_lev_med[j]))
    probs <- (counts + 1) / (length(idx) + num_lev_med[j])
    prob_mat[k, ] <- as.numeric(probs)
  }
  init_log_probs_med[[j]] <- log(prob_mat)
}

res_med_1iter <- run_deterministic_kamila(
  con_mat = con_mat_med,
  cat_mat = cat_mat_med,
  num_clust = k2,
  init_means = init_means_med,
  init_log_probs = init_log_probs_med,
  num_lev = num_lev_med,
  con_weights = con_weights_med,
  cat_weights = cat_weights_med,
  cat_bw = cat_bw_med,
  max_iter = 1L
)

res_med_conv <- run_deterministic_kamila(
  con_mat = con_mat_med,
  cat_mat = cat_mat_med,
  num_clust = k2,
  init_means = init_means_med,
  init_log_probs = init_log_probs_med,
  num_lev = num_lev_med,
  con_weights = con_weights_med,
  cat_weights = cat_weights_med,
  cat_bw = cat_bw_med,
  max_iter = 25L
)

data_med <- list(
  description = "Medium mixed dataset with 60 observations, 3 continuous vars, 2 categorical vars (2 & 3 levels), K=3",
  n_samples = n2,
  n_con = 3,
  n_cat = 2,
  num_clust = k2,
  num_lev = num_lev_med,
  con_weights = con_weights_med,
  cat_weights = cat_weights_med,
  cat_bw = cat_bw_med,
  con_data = con_mat_med,
  cat_data = cat_mat_med - 1L,
  init_means = init_means_med,
  init_log_probs = init_log_probs_med,
  one_iteration = list(
    final_membership = as.numeric(res_med_1iter$finalMemb) - 1L,
    final_means = res_med_1iter$finalMeans,
    final_log_probs = res_med_1iter$finalLogProbs,
    total_log_lik = res_med_1iter$totalLogLik,
    cat_log_lik = res_med_1iter$catLogLik,
    win_dist = res_med_1iter$winDist,
    num_iter = res_med_1iter$numIter
  ),
  converged = list(
    final_membership = as.numeric(res_med_conv$finalMemb) - 1L,
    final_means = res_med_conv$finalMeans,
    final_log_probs = res_med_conv$finalLogProbs,
    total_log_lik = res_med_conv$totalLogLik,
    cat_log_lik = res_med_conv$catLogLik,
    win_dist = res_med_conv$winDist,
    num_iter = res_med_conv$numIter
  )
)

write_json(data_med, file.path(out_dir, "reference_medium_mixed.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10)

# -----------------------------------------------------------------------------
# 3. Dataset 3: Continuous-Only Data (N=30, P_con=2, K=2)
# -----------------------------------------------------------------------------
cat("Generating Dataset 3 (Continuous Only)...\n")
n3 <- 30
k3 <- 2L
con_mat_only <- rbind(
  matrix(rnorm(15 * 2, mean = 2.0, sd = 0.4), nrow = 15, ncol = 2),
  matrix(rnorm(15 * 2, mean = 7.0, sd = 0.5), nrow = 15, ncol = 2)
)
con_weights_only <- c(1.0, 1.0)
init_means_only <- matrix(c(2.0, 2.0, 7.0, 7.0), nrow = k3, ncol = 2, byrow = TRUE)

res_con_1iter <- run_deterministic_kamila(
  con_mat = con_mat_only,
  cat_mat = NULL,
  num_clust = k3,
  init_means = init_means_only,
  init_log_probs = list(),
  num_lev = integer(0),
  con_weights = con_weights_only,
  cat_weights = numeric(0),
  cat_bw = 0.025,
  max_iter = 1L,
  has_con = TRUE,
  has_cat = FALSE
)

res_con_conv <- run_deterministic_kamila(
  con_mat = con_mat_only,
  cat_mat = NULL,
  num_clust = k3,
  init_means = init_means_only,
  init_log_probs = list(),
  num_lev = integer(0),
  con_weights = con_weights_only,
  cat_weights = numeric(0),
  cat_bw = 0.025,
  max_iter = 25L,
  has_con = TRUE,
  has_cat = FALSE
)

data_con_only <- list(
  description = "Continuous-only dataset with 30 observations, 2 continuous vars, K=2",
  n_samples = n3,
  n_con = 2,
  n_cat = 0,
  num_clust = k3,
  con_weights = con_weights_only,
  con_data = con_mat_only,
  init_means = init_means_only,
  one_iteration = list(
    final_membership = as.numeric(res_con_1iter$finalMemb) - 1L,
    final_means = res_con_1iter$finalMeans,
    total_log_lik = res_con_1iter$totalLogLik,
    win_dist = res_con_1iter$winDist,
    num_iter = res_con_1iter$numIter
  ),
  converged = list(
    final_membership = as.numeric(res_con_conv$finalMemb) - 1L,
    final_means = res_con_conv$finalMeans,
    total_log_lik = res_con_conv$totalLogLik,
    win_dist = res_con_conv$winDist,
    num_iter = res_con_conv$numIter
  )
)

write_json(data_con_only, file.path(out_dir, "reference_continuous_only.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10)

# -----------------------------------------------------------------------------
# 4. Dataset 4: Categorical-Only Data (N=30, P_cat=2, K=2)
# -----------------------------------------------------------------------------
cat("Generating Dataset 4 (Categorical Only)...\n")
n4 <- 30
k4 <- 2L
cat_mat_only <- rbind(
  cbind(sample(1:2, 15, replace = TRUE, prob = c(0.85, 0.15)), sample(1:2, 15, replace = TRUE, prob = c(0.85, 0.15))),
  cbind(sample(1:2, 15, replace = TRUE, prob = c(0.15, 0.85)), sample(1:2, 15, replace = TRUE, prob = c(0.15, 0.85)))
)
num_lev_cat_only <- as.integer(c(2, 2))
cat_weights_only <- c(1.0, 1.0)
init_lp_cat_only <- list(
  matrix(log(c(0.8, 0.2, 0.2, 0.8)), nrow = k4, ncol = 2, byrow = TRUE),
  matrix(log(c(0.8, 0.2, 0.2, 0.8)), nrow = k4, ncol = 2, byrow = TRUE)
)

res_cat_1iter <- run_deterministic_kamila(
  con_mat = NULL,
  cat_mat = cat_mat_only,
  num_clust = k4,
  init_means = NULL,
  init_log_probs = init_lp_cat_only,
  num_lev = num_lev_cat_only,
  con_weights = numeric(0),
  cat_weights = cat_weights_only,
  cat_bw = 0.025,
  max_iter = 1L,
  has_con = FALSE,
  has_cat = TRUE
)

res_cat_conv <- run_deterministic_kamila(
  con_mat = NULL,
  cat_mat = cat_mat_only,
  num_clust = k4,
  init_means = NULL,
  init_log_probs = init_lp_cat_only,
  num_lev = num_lev_cat_only,
  con_weights = numeric(0),
  cat_weights = cat_weights_only,
  cat_bw = 0.025,
  max_iter = 25L,
  has_con = FALSE,
  has_cat = TRUE
)

data_cat_only <- list(
  description = "Categorical-only dataset with 30 observations, 2 categorical vars (2 levels each), K=2",
  n_samples = n4,
  n_con = 0,
  n_cat = 2,
  num_clust = k4,
  num_lev = num_lev_cat_only,
  cat_weights = cat_weights_only,
  cat_bw = 0.025,
  cat_data = cat_mat_only - 1L,
  init_log_probs = init_lp_cat_only,
  one_iteration = list(
    final_membership = as.numeric(res_cat_1iter$finalMemb) - 1L,
    final_log_probs = res_cat_1iter$finalLogProbs,
    cat_log_lik = res_cat_1iter$catLogLik,
    num_iter = res_cat_1iter$numIter
  ),
  converged = list(
    final_membership = as.numeric(res_cat_conv$finalMemb) - 1L,
    final_log_probs = res_cat_conv$finalLogProbs,
    cat_log_lik = res_cat_conv$catLogLik,
    num_iter = res_cat_conv$numIter
  )
)

write_json(data_cat_only, file.path(out_dir, "reference_categorical_only.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("Reference datasets successfully generated in:", out_dir, "\n")
