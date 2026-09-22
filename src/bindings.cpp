#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "kamila_core.hpp"

namespace nb = nanobind;
using namespace nb::literals;

static std::vector<double> extract_double_vector(nb::handle item) {
    if (nb::isinstance<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(item)) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(item);
        return std::vector<double>(arr.data(), arr.data() + arr.size());
    }
    if (nb::isinstance<nb::ndarray<const double, nb::device::cpu>>(item)) {
        auto arr = nb::cast<nb::ndarray<const double, nb::device::cpu>>(item);
        return std::vector<double>(arr.data(), arr.data() + arr.size());
    }
    std::vector<double> vals;
    if (nb::isinstance<nb::sequence>(item)) {
        for (auto sub_item : nb::cast<nb::sequence>(item)) {
            if (nb::isinstance<nb::sequence>(sub_item) && !nb::isinstance<nb::str>(sub_item)) {
                for (auto val : nb::cast<nb::sequence>(sub_item)) {
                    vals.push_back(nb::cast<double>(val));
                }
            } else {
                vals.push_back(nb::cast<double>(sub_item));
            }
        }
    }
    return vals;
}

nb::dict kamila_loop_cpp(
    nb::object con_data_obj,
    nb::object cat_data_obj,
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    nb::object con_weights_obj,
    nb::object cat_weights_obj,
    nb::object num_levels_obj,
    nb::object init_means_obj,
    nb::object init_log_probs_obj,
    double cat_bw,
    int max_iter,
    bool has_con,
    bool has_cat
) {
    const double* con_ptr = nullptr;
    if (has_con && !con_data_obj.is_none()) {
        auto con_arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(con_data_obj);
        con_ptr = con_arr.data();
    }

    const int* cat_ptr = nullptr;
    if (has_cat && !cat_data_obj.is_none()) {
        auto cat_arr = nb::cast<nb::ndarray<const int, nb::c_contig, nb::device::cpu>>(cat_data_obj);
        cat_ptr = cat_arr.data();
    }

    const double* con_wgts_ptr = nullptr;
    if (has_con && !con_weights_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(con_weights_obj);
        con_wgts_ptr = arr.data();
    }

    const double* cat_wgts_ptr = nullptr;
    if (has_cat && !cat_weights_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(cat_weights_obj);
        cat_wgts_ptr = arr.data();
    }

    const int* num_levels_ptr = nullptr;
    if (has_cat && !num_levels_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const int, nb::c_contig, nb::device::cpu>>(num_levels_obj);
        num_levels_ptr = arr.data();
    }

    const double* init_means_ptr = nullptr;
    if (has_con && !init_means_obj.is_none()) {
        auto means_arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(init_means_obj);
        init_means_ptr = means_arr.data();
    }

    std::vector<std::vector<double>> init_log_probs;
    if (has_cat && !init_log_probs_obj.is_none()) {
        for (auto item : nb::cast<nb::sequence>(init_log_probs_obj)) {
            init_log_probs.push_back(extract_double_vector(item));
        }
    }

    kamila::KamilaResult res = kamila::kamila_loop(
        con_ptr,
        cat_ptr,
        n_samples,
        n_con,
        n_cat,
        n_clusters,
        con_wgts_ptr,
        cat_wgts_ptr,
        num_levels_ptr,
        init_means_ptr,
        init_log_probs,
        cat_bw,
        max_iter,
        has_con,
        has_cat
    );

    nb::dict out;
    out["num_iter"] = res.num_iter;
    out["degenerate_soln"] = res.degenerate_soln;
    out["final_membership"] = res.final_membership;
    out["final_means"] = res.final_means;
    out["final_log_probs"] = res.final_log_probs;
    out["total_log_lik"] = res.total_log_lik;
    out["cat_log_lik"] = res.cat_log_lik;
    out["win_dist"] = res.win_dist;
    out["total_dist"] = res.total_dist;
    out["objective"] = res.objective;

    return out;
}

std::vector<int> kamila_predict_cpp(
    nb::object con_data_obj,
    nb::object cat_data_obj,
    int n_samples,
    int n_con,
    int n_cat,
    int n_clusters,
    nb::object con_weights_obj,
    nb::object cat_weights_obj,
    nb::object fitted_means_obj,
    nb::object fitted_log_probs_obj,
    bool has_con,
    bool has_cat
) {
    const double* con_ptr = nullptr;
    if (has_con && !con_data_obj.is_none()) {
        auto con_arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(con_data_obj);
        con_ptr = con_arr.data();
    }

    const int* cat_ptr = nullptr;
    if (has_cat && !cat_data_obj.is_none()) {
        auto cat_arr = nb::cast<nb::ndarray<const int, nb::c_contig, nb::device::cpu>>(cat_data_obj);
        cat_ptr = cat_arr.data();
    }

    const double* con_wgts_ptr = nullptr;
    if (has_con && !con_weights_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(con_weights_obj);
        con_wgts_ptr = arr.data();
    }

    const double* cat_wgts_ptr = nullptr;
    if (has_cat && !cat_weights_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(cat_weights_obj);
        cat_wgts_ptr = arr.data();
    }

    const double* fitted_means_ptr = nullptr;
    if (has_con && !fitted_means_obj.is_none()) {
        auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(fitted_means_obj);
        fitted_means_ptr = arr.data();
    }

    std::vector<std::vector<double>> fitted_log_probs;
    if (has_cat && !fitted_log_probs_obj.is_none()) {
        for (auto item : nb::cast<nb::sequence>(fitted_log_probs_obj)) {
            fitted_log_probs.push_back(extract_double_vector(item));
        }
    }

    return kamila::kamila_predict(
        con_ptr,
        cat_ptr,
        n_samples,
        n_con,
        n_cat,
        n_clusters,
        con_wgts_ptr,
        cat_wgts_ptr,
        fitted_means_ptr,
        fitted_log_probs,
        nullptr,
        0,
        has_con,
        has_cat
    );
}

int get_cpp_version() {
    return 1;
}

NB_MODULE(_kamila_cpp, m) {
    m.doc() = "C++ accelerated core routines for scikit-kamila using nanobind";
    m.def("get_cpp_version", &get_cpp_version, "Return the C++ engine version.");
    m.def(
        "kamila_loop_cpp",
        &kamila_loop_cpp,
        "con_data"_a.none() = nb::none(),
        "cat_data"_a.none() = nb::none(),
        "n_samples"_a,
        "n_con"_a,
        "n_cat"_a,
        "n_clusters"_a,
        "con_weights"_a.none() = nb::none(),
        "cat_weights"_a.none() = nb::none(),
        "num_levels"_a.none() = nb::none(),
        "init_means"_a.none() = nb::none(),
        "init_log_probs"_a.none() = nb::none(),
        "cat_bw"_a,
        "max_iter"_a,
        "has_con"_a,
        "has_cat"_a,
        "Run core iterative KAMILA algorithm."
    );
    m.def(
        "kamila_predict_cpp",
        &kamila_predict_cpp,
        "con_data"_a.none() = nb::none(),
        "cat_data"_a.none() = nb::none(),
        "n_samples"_a,
        "n_con"_a,
        "n_cat"_a,
        "n_clusters"_a,
        "con_weights"_a.none() = nb::none(),
        "cat_weights"_a.none() = nb::none(),
        "fitted_means"_a.none() = nb::none(),
        "fitted_log_probs"_a.none() = nb::none(),
        "has_con"_a,
        "has_cat"_a,
        "Predict cluster memberships for new observations."
    );
}
