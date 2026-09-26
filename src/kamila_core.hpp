#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

namespace kamila {

struct KamilaResult {
    int num_iter = 0;
    bool degenerate_soln = false;
    std::vector<int> final_membership;              // 0-indexed cluster labels of shape (N,)
    std::vector<double> final_means;                // shape (K, P_con) row-major
    std::vector<std::vector<double>> final_log_probs;// list of Q matrices, each (K, n_levels[q]) row-major
    double total_log_lik = 0.0;
    double cat_log_lik = 0.0;
    double win_dist = 0.0;
    double total_dist = 0.0;
    double objective = 0.0;
    std::vector<double> final_min_dist;             // shape (N,) minimum continuous distance to final_means
};

// Core iterative loop function matching Rcpp kamilaLoopCpp
KamilaResult kamila_loop(
    const double* con_data,             // N x P_con (row-major) or nullptr if has_con == false
    const int* cat_data,                // N x P_cat (row-major, 0-indexed) or nullptr if has_cat == false
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    const double* con_weights,          // length P_con
    const double* cat_weights,          // length P_cat
    const int* num_levels,              // length P_cat (number of categories per variable)
    const double* init_means,           // K x P_con (row-major) or nullptr
    const std::vector<std::vector<double>>& init_log_probs, // list of Q matrices, each K x n_levels[q]
    double cat_bw,
    int max_iter,
    bool has_con,
    bool has_cat,
    std::uint64_t seed                  // seeds re-initialization of empty clusters
);

// Predict cluster memberships for new observations
std::vector<int> kamila_predict(
    const double* con_data,             // N x P_con (row-major) or nullptr
    const int* cat_data,                // N x P_cat (row-major, 0-indexed) or nullptr
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    const double* con_weights,
    const double* cat_weights,
    const double* fitted_means,         // K x P_con (row-major)
    const std::vector<std::vector<double>>& fitted_log_probs,
    const double* all_data_min_dist,    // shape (N_ref,) minimum continuous distance or nullptr
    int n_ref_samples,
    bool has_con,
    bool has_cat
);

} // namespace kamila
