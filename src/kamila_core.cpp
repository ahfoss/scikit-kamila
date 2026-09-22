#include "kamila_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#ifndef M_1_SQRT_2PI
#define M_1_SQRT_2PI 0.398942280401432677939946059934
#endif

namespace kamila {

KamilaResult kamila_loop(
    const double* con_data,
    const int* cat_data,
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    const double* con_weights,
    const double* cat_weights,
    const int* num_levels,
    const double* init_means,
    const std::vector<std::vector<double>>& init_log_probs,
    double cat_bw,
    int max_iter,
    bool has_con,
    bool has_cat
) {
    KamilaResult result;
    int nn = n_samples;
    int pp = has_con ? n_con : 0;
    int qq = has_cat ? n_cat : 0;
    int kk = n_clusters;

    // Allocate scratch buffers
    std::vector<double> dist_mat(has_con ? (nn * kk) : 0, 0.0);
    std::vector<double> min_dist(has_con ? nn : 0, 0.0);
    std::vector<double> cat_log_liks(has_cat ? (nn * kk) : 0, 0.0);
    std::vector<double> all_log_liks(nn * kk, 0.0);
    std::vector<int> memb_old(nn, -1);
    std::vector<int> memb_new(nn, 0);
    std::vector<double> count_vec(kk, 0.0);

    // Current means: shape (kk, pp) row-major
    std::vector<double> current_means(has_con ? (kk * pp) : 0, 0.0);
    if (has_con && init_means != nullptr) {
        std::copy(init_means, init_means + kk * pp, current_means.begin());
    }

    // Current log probs: list of Q matrices, each shape (kk, num_levels[q]) row-major
    std::vector<std::vector<double>> log_probs(qq);
    if (has_cat) {
        for (int q = 0; q < qq; ++q) {
            int nlev = num_levels[q];
            log_probs[q].resize(kk * nlev);
            if (q < static_cast<int>(init_log_probs.size()) && !init_log_probs[q].empty()) {
                std::copy(init_log_probs[q].begin(), init_log_probs[q].begin() + kk * nlev, log_probs[q].begin());
            }
        }
    }

    // Compute total continuous distance T to overall mean
    double total_dist = 0.0;
    if (has_con) {
        std::vector<double> grand_mean(pp, 0.0);
        for (int i = 0; i < nn; ++i) {
            for (int p = 0; p < pp; ++p) {
                grand_mean[p] += con_data[i * pp + p];
            }
        }
        for (int p = 0; p < pp; ++p) {
            grand_mean[p] /= nn;
        }

        for (int i = 0; i < nn; ++i) {
            double sum_sq = 0.0;
            for (int p = 0; p < pp; ++p) {
                double w = con_weights[p];
                if (w == 0.0) continue;
                double diff = con_data[i * pp + p] - grand_mean[p];
                if (w == 1.0) {
                    sum_sq += diff * diff;
                } else {
                    sum_sq += (w * diff) * (w * diff);
                }
            }
            total_dist += std::sqrt(sum_sq);
        }
    }

    // KDE workspace
    const int m_grid = 401;
    std::vector<double> r_sorted(has_con ? nn : 0);
    std::vector<double> gcounts(m_grid);
    std::vector<double> kappa(m_grid + 1);
    std::vector<double> y_kde(m_grid);
    std::vector<double> new_y(m_grid);
    std::vector<double> rad_y(m_grid);
    std::vector<double> dens_r(m_grid);
    std::vector<double> diff_dens_r(m_grid - 1);

    // Categorical workspace
    int max_nlev = 0;
    for (int q = 0; q < qq; ++q) {
        if (num_levels[q] > max_nlev) max_nlev = num_levels[q];
    }
    std::vector<int> raw_tab(kk * max_nlev);
    std::vector<double> out_mat(kk * max_nlev);
    std::vector<double> mid_mat(kk * max_nlev);
    std::vector<int> col_sums(max_nlev);
    std::vector<double> row_sums(kk);
    std::vector<double> final_row_sums(kk);

    int num_iter = 0;
    bool degenerate_soln = false;

    while (true) {
        bool can_stop = (num_iter >= 3);
        if (can_stop) {
            bool all_equal = true;
            for (int i = 0; i < nn; ++i) {
                if (memb_old[i] != memb_new[i]) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) break;
        }
        if (num_iter >= max_iter) break;

        num_iter++;

        // 1. Continuous calculations: distances and radial KDE
        if (has_con) {
            double max_eval = 0.0;
            for (int i = 0; i < nn; ++i) {
                double min_d = std::numeric_limits<double>::infinity();
                for (int j = 0; j < kk; ++j) {
                    double sum_sq = 0.0;
                    for (int p = 0; p < pp; ++p) {
                        double w = con_weights[p];
                        if (w == 0.0) continue;
                        double diff = con_data[i * pp + p] - current_means[j * pp + p];
                        if (w == 1.0) {
                            sum_sq += diff * diff;
                        } else {
                            sum_sq += (w * diff) * (w * diff);
                        }
                    }
                    double d = std::sqrt(sum_sq);
                    dist_mat[i * kk + j] = d;
                    if (d < min_d) min_d = d;
                    if (d > max_eval) max_eval = d;
                }
                min_dist[i] = min_d;
            }

            // Radial KDE on minDist: bw.nrd0 bandwidth estimation
            double mean_r = 0.0;
            for (int i = 0; i < nn; ++i) {
                r_sorted[i] = min_dist[i];
                mean_r += min_dist[i];
            }
            mean_r /= nn;

            double var_r = 0.0;
            for (int i = 0; i < nn; ++i) {
                double diff = min_dist[i] - mean_r;
                var_r += diff * diff;
            }
            double hi = (nn > 1) ? std::sqrt(var_r / (nn - 1.0)) : 0.0;

            // Type 7 quantile using std::nth_element
            double index25 = 1.0 + (nn - 1.0) * 0.25;
            int lo25 = std::max(0, std::min(nn - 1, static_cast<int>(std::floor(index25)) - 1));
            int hi25 = std::max(0, std::min(nn - 1, static_cast<int>(std::ceil(index25)) - 1));
            double g25 = index25 - std::floor(index25);

            double index75 = 1.0 + (nn - 1.0) * 0.75;
            int lo75 = std::max(0, std::min(nn - 1, static_cast<int>(std::floor(index75)) - 1));
            int hi75 = std::max(0, std::min(nn - 1, static_cast<int>(std::ceil(index75)) - 1));
            double g75 = index75 - std::floor(index75);

            std::nth_element(r_sorted.begin(), r_sorted.begin() + lo25, r_sorted.end());
            double v_lo25 = r_sorted[lo25];
            std::nth_element(r_sorted.begin() + lo25 + 1, r_sorted.begin() + hi25, r_sorted.end());
            double v_hi25 = r_sorted[hi25];
            double q25 = (lo25 == hi25) ? v_lo25 : ((1.0 - g25) * v_lo25 + g25 * v_hi25);

            std::nth_element(r_sorted.begin() + hi25 + 1, r_sorted.begin() + lo75, r_sorted.end());
            double v_lo75 = r_sorted[lo75];
            std::nth_element(r_sorted.begin() + lo75 + 1, r_sorted.begin() + hi75, r_sorted.end());
            double v_hi75 = r_sorted[hi75];
            double q75 = (lo75 == hi75) ? v_lo75 : ((1.0 - g75) * v_lo75 + g75 * v_hi75);
            double iqr = q75 - q25;

            double lo = std::min(hi, iqr / 1.34);
            if (lo <= 0.0 || std::isnan(lo)) {
                lo = hi;
                if (lo <= 0.0 || std::isnan(lo)) {
                    lo = std::abs(min_dist[0]);
                    if (lo <= 0.0 || std::isnan(lo)) {
                        lo = 1.0;
                    }
                }
            }
            double h_bw = 0.9 * lo * std::pow(static_cast<double>(nn), -0.2);

            // Linear binning into gcounts
            double delta_grid = (max_eval > 0.0) ? (max_eval / (m_grid - 1.0)) : 1.0;
            double inv_delta = (delta_grid > 0.0) ? (1.0 / delta_grid) : 0.0;
            std::fill(gcounts.begin(), gcounts.end(), 0.0);

            for (int i = 0; i < nn; ++i) {
                double r = min_dist[i];
                if (r >= 0.0 && r < max_eval) {
                    double pos = r * inv_delta;
                    int l = static_cast<int>(pos);
                    double rem = pos - l;
                    if (l >= 0 && l < m_grid - 1) {
                        gcounts[l] += (1.0 - rem);
                        gcounts[l + 1] += rem;
                    }
                }
            }

            // Discrete Gaussian convolution
            double delta = delta_grid / h_bw;
            int L = (delta > 0.0) ? static_cast<int>(std::floor(4.0 / delta)) : m_grid;
            if (L > m_grid) L = m_grid;
            if (L < 0) L = 0;

            double sum_kappa = 0.0;
            for (int l = 0; l <= L; ++l) {
                double z = l * delta;
                kappa[l] = std::exp(-0.5 * z * z) * M_1_SQRT_2PI / (nn * h_bw);
                sum_kappa += (l == 0) ? kappa[l] : (2.0 * kappa[l]);
            }
            double tot = sum_kappa * delta_grid * nn;
            double inv_tot = (tot > 0.0) ? (1.0 / tot) : 0.0;
            for (int l = 0; l <= L; ++l) {
                kappa[l] *= inv_tot;
            }

            for (int i = 0; i < m_grid; ++i) {
                int j_min = std::max(0, i - L);
                int j_max = std::min(m_grid - 1, i + L);
                double val = 0.0;
                for (int j = j_min; j <= j_max; ++j) {
                    val += gcounts[j] * kappa[std::abs(i - j)];
                }
                y_kde[i] = val;
            }

            // Radial density post-processing
            double min_pos = 1e300;
            for (int i = 0; i < m_grid; ++i) {
                if (y_kde[i] > 0.0 && y_kde[i] < min_pos) min_pos = y_kde[i];
            }
            for (int i = 0; i < m_grid; ++i) {
                new_y[i] = (y_kde[i] > 0.0) ? y_kde[i] : (min_pos / 100.0);
            }

            double x19 = 19.0 * delta_grid;
            double slope = (x19 > 0.0) ? (new_y[19] / x19) : 0.0;
            for (int i = 0; i < 20; ++i) {
                new_y[i] = (i * delta_grid) * slope;
            }

            for (int i = 1; i < m_grid; ++i) {
                double xi = i * delta_grid;
                rad_y[i] = new_y[i] / std::pow(xi, pp - 1);
            }
            rad_y[0] = rad_y[1];

            double sum_rad_y = 0.0;
            for (int i = 0; i < m_grid; ++i) {
                if (rad_y[i] > 1.0) rad_y[i] = 1.0;
                sum_rad_y += rad_y[i];
            }

            double min_dens_r = 1e300;
            double norm_factor = delta_grid * sum_rad_y;
            for (int i = 0; i < m_grid; ++i) {
                dens_r[i] = (norm_factor > 0.0) ? (rad_y[i] / norm_factor) : 0.0;
                if (dens_r[i] < min_dens_r) min_dens_r = dens_r[i];
            }

            for (int i = 0; i < m_grid - 1; ++i) {
                diff_dens_r[i] = dens_r[i + 1] - dens_r[i];
            }

            // Interpolate continuous log densities directly into all_log_liks
            double log_val0 = std::log((dens_r[0] > min_dens_r) ? dens_r[0] : min_dens_r);
            double log_val_max = std::log((dens_r[m_grid - 1] > min_dens_r) ? dens_r[m_grid - 1] : min_dens_r);

            for (int i = 0; i < nn; ++i) {
                for (int j = 0; j < kk; ++j) {
                    double u = dist_mat[i * kk + j];
                    if (u <= 0.0) {
                        all_log_liks[i * kk + j] = log_val0;
                    } else if (u >= max_eval) {
                        all_log_liks[i * kk + j] = log_val_max;
                    } else {
                        double pos = u * inv_delta;
                        int idx = static_cast<int>(pos);
                        if (idx >= m_grid - 1) idx = m_grid - 2;
                        double frac = pos - idx;
                        double val = dens_r[idx] + frac * diff_dens_r[idx];
                        all_log_liks[i * kk + j] = std::log((val > min_dens_r) ? val : min_dens_r);
                    }
                }
            }
        }

        // 2. Categorical calculations
        if (has_cat) {
            std::fill(cat_log_liks.begin(), cat_log_liks.end(), 0.0);

            for (int i = 0; i < nn; ++i) {
                for (int j = 0; j < kk; ++j) {
                    double cat_ll = 0.0;
                    for (int q = 0; q < qq; ++q) {
                        double w = cat_weights[q];
                        if (w == 0.0) continue;
                        int lev = cat_data[i * qq + q];
                        int nlev = num_levels[q];
                        if (lev >= 0 && lev < nlev) {
                            cat_ll += w * log_probs[q][j * nlev + lev];
                        }
                    }
                    cat_log_liks[i * kk + j] = cat_ll;
                    if (has_con) {
                        all_log_liks[i * kk + j] += cat_ll;
                    } else {
                        all_log_liks[i * kk + j] = cat_ll;
                    }
                }
            }
        }

        // 3. Partition data into clusters: memb_old <- memb_new, memb_new <- argmax
        std::copy(memb_new.begin(), memb_new.end(), memb_old.begin());
        std::fill(count_vec.begin(), count_vec.end(), 0.0);

        for (int i = 0; i < nn; ++i) {
            double max_val = all_log_liks[i * kk + 0];
            int max_idx = 0;
            for (int j = 1; j < kk; ++j) {
                double val = all_log_liks[i * kk + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            memb_new[i] = max_idx;
            count_vec[max_idx] += 1.0;
        }

        // 4. Update continuous means
        if (has_con) {
            std::fill(current_means.begin(), current_means.end(), 0.0);
            for (int i = 0; i < nn; ++i) {
                int cl = memb_new[i];
                for (int p = 0; p < pp; ++p) {
                    current_means[cl * pp + p] += con_data[i * pp + p];
                }
            }
            for (int j = 0; j < kk; ++j) {
                if (count_vec[j] > 0.0) {
                    for (int p = 0; p < pp; ++p) {
                        current_means[j * pp + p] /= count_vec[j];
                    }
                }
            }
        }

        // 5. Update categorical probabilities with 2D smoothing
        if (has_cat) {
            for (int q = 0; q < qq; ++q) {
                int nlev = num_levels[q];

                // (a) Tabulate
                std::fill(raw_tab.begin(), raw_tab.begin() + kk * nlev, 0);
                for (int i = 0; i < nn; ++i) {
                    int cl = memb_new[i];
                    int lev = cat_data[i * qq + q];
                    if (cl >= 0 && cl < kk && lev >= 0 && lev < nlev) {
                        raw_tab[cl * nlev + lev] += 1;
                    }
                }

                // (b) Smooth
                if (cat_bw != 0.0) {
                    std::fill(col_sums.begin(), col_sums.begin() + nlev, 0);
                    for (int j = 0; j < nlev; ++j) {
                        for (int i = 0; i < kk; ++i) {
                            col_sums[j] += raw_tab[i * nlev + j];
                        }
                    }

                    double bw_div_k = (kk > 1) ? (cat_bw / (kk - 1.0)) : 0.0;
                    double one_minus_bw = 1.0 - cat_bw;
                    for (int i = 0; i < kk; ++i) {
                        for (int j = 0; j < nlev; ++j) {
                            int off_counts1 = col_sums[j] - raw_tab[i * nlev + j];
                            mid_mat[i * nlev + j] = one_minus_bw * raw_tab[i * nlev + j] + bw_div_k * off_counts1;
                        }
                    }

                    std::fill(row_sums.begin(), row_sums.begin() + kk, 0.0);
                    for (int i = 0; i < kk; ++i) {
                        for (int j = 0; j < nlev; ++j) {
                            row_sums[i] += mid_mat[i * nlev + j];
                        }
                    }

                    double bw_div_lev = (nlev > 1) ? (cat_bw / (nlev - 1.0)) : 0.0;
                    for (int i = 0; i < kk; ++i) {
                        for (int j = 0; j < nlev; ++j) {
                            double off_counts2 = row_sums[i] - mid_mat[i * nlev + j];
                            out_mat[i * nlev + j] = one_minus_bw * mid_mat[i * nlev + j] + bw_div_lev * off_counts2;
                        }
                    }
                } else {
                    for (int idx = 0; idx < kk * nlev; ++idx) {
                        out_mat[idx] = static_cast<double>(raw_tab[idx]);
                    }
                }

                // (c) Normalize and take log
                std::fill(final_row_sums.begin(), final_row_sums.begin() + kk, 0.0);
                for (int i = 0; i < kk; ++i) {
                    for (int j = 0; j < nlev; ++j) {
                        final_row_sums[i] += out_mat[i * nlev + j];
                    }
                }

                for (int i = 0; i < kk; ++i) {
                    double denom = final_row_sums[i];
                    for (int j = 0; j < nlev; ++j) {
                        log_probs[q][i * nlev + j] = (denom > 0.0 && out_mat[i * nlev + j] > 0.0)
                            ? std::log(out_mat[i * nlev + j] / denom)
                            : -std::numeric_limits<double>::infinity();
                    }
                }
            }
        }

        // Check for degenerate solution (empty cluster)
        for (int j = 0; j < kk; ++j) {
            if (count_vec[j] == 0.0) {
                degenerate_soln = true;
                break;
            }
        }
        if (degenerate_soln) break;
    }

    // Compute summary statistics
    double total_log_lik = 0.0;
    if (degenerate_soln) {
        total_log_lik = -std::numeric_limits<double>::infinity();
    } else {
        for (int i = 0; i < nn; ++i) {
            double max_val = all_log_liks[i * kk + 0];
            for (int j = 1; j < kk; ++j) {
                double val = all_log_liks[i * kk + j];
                if (val > max_val) max_val = val;
            }
            total_log_lik += max_val;
        }
    }

    double cat_log_lik = 0.0;
    if (has_cat) {
        for (int i = 0; i < nn; ++i) {
            double max_val = cat_log_liks[i * kk + 0];
            for (int j = 1; j < kk; ++j) {
                double val = cat_log_liks[i * kk + j];
                if (val > max_val) max_val = val;
            }
            cat_log_lik += max_val;
        }
    }

    double win_dist = 0.0;
    if (has_con) {
        for (int i = 0; i < nn; ++i) {
            int cl = memb_new[i];
            win_dist += dist_mat[i * kk + cl];
        }
    }

    // Objective value calculation matching R
    double objective = 0.0;
    if (has_con && has_cat) {
        double denom = total_dist - win_dist;
        double win_to_bet_rat = (denom > 1e-12) ? (win_dist / denom) : 100.0;
        if (win_to_bet_rat < 0.0 || std::isnan(win_to_bet_rat) || std::isinf(win_to_bet_rat)) {
            win_to_bet_rat = 100.0;
        }
        objective = (cat_log_lik == 0.0) ? 0.0 : (win_to_bet_rat * cat_log_lik);
    } else if (has_con) {
        objective = total_log_lik;
    } else {
        objective = degenerate_soln ? -std::numeric_limits<double>::infinity() : cat_log_lik;
    }

    result.num_iter = num_iter;
    result.degenerate_soln = degenerate_soln;
    result.final_membership = memb_new;
    result.final_means = current_means;
    result.final_log_probs = log_probs;
    result.total_log_lik = total_log_lik;
    result.cat_log_lik = cat_log_lik;
    result.win_dist = win_dist;
    result.total_dist = total_dist;
    result.objective = objective;

    return result;
}

std::vector<int> kamila_predict(
    const double* con_data,
    const int* cat_data,
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    const double* con_weights,
    const double* cat_weights,
    const double* fitted_means,
    const std::vector<std::vector<double>>& fitted_log_probs,
    bool has_con,
    bool has_cat
) {
    int nn = n_samples;
    int pp = has_con ? n_con : 0;
    int qq = has_cat ? n_cat : 0;
    int kk = n_clusters;

    std::vector<int> predictions(nn, 0);

    for (int i = 0; i < nn; ++i) {
        double best_score = -std::numeric_limits<double>::infinity();
        int best_cluster = 0;

        for (int k = 0; k < kk; ++k) {
            double score = 0.0;

            if (has_con) {
                // Continuous distance penalty (negative distance)
                double sum_sq = 0.0;
                for (int p = 0; p < pp; ++p) {
                    double w = con_weights[p];
                    if (w == 0.0) continue;
                    double diff = con_data[i * pp + p] - fitted_means[k * pp + p];
                    if (w == 1.0) {
                        sum_sq += diff * diff;
                    } else {
                        sum_sq += (w * diff) * (w * diff);
                    }
                }
                score -= std::sqrt(sum_sq);
            }

            if (has_cat) {
                // Categorical log likelihood
                for (int q = 0; q < qq; ++q) {
                    double w = cat_weights[q];
                    if (w == 0.0) continue;
                    int lev = cat_data[i * qq + q];
                    int nlev = static_cast<int>(fitted_log_probs[q].size() / kk);
                    if (lev >= 0 && lev < nlev) {
                        score += w * fitted_log_probs[q][k * nlev + lev];
                    }
                }
            }

            if (score > best_score || k == 0) {
                best_score = score;
                best_cluster = k;
            }
        }
        predictions[i] = best_cluster;
    }

    return predictions;
}

} // namespace kamila
