#include <Rcpp.h>
#include "screening.h"
#include "cfaad/AADInit.hpp"
#include <string>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <cmath>

using namespace Rcpp;


// -------------------------------------------------------------
// Helper functions to safely extract list elements
// -------------------------------------------------------------
inline double get_d(const List& L, const char* name, double def = 0.0) {
  if (L.containsElementNamed(name)) return as<double>(L[name]);
  return def;
}

inline std::string get_s(const List& L, const char* name, std::string def = "") {
  if (L.containsElementNamed(name)) return as<std::string>(L[name]);
  return def;
}

// -------------------------------------------------------------
// Unified Rcpp Export Function
// -------------------------------------------------------------

//' @importFrom Rcpp sourceCpp 
//' Compute Likelihoods for Screening Models
//'
//' @name screening_model_likes
//' @param model integer; one of 1..5
//' @param inputs list; per-subject screening histories
//' @param onset_pars list; onset distribution parameters
//' @param sojourn_pars list; sojourn distribution parameters
//' @param biom_pars list; biomarker / test parameters
//' @param grads logical; if TRUE return AD gradients
//' @param tol numeric; integration tolerance
//' @param return_type character; "" for raw likelihoods, "weighted_ll" for weighted log-likelihood
//' @param weights numeric vector or NULL
//' @param left_trunc logical; if TRUE apply left-truncation correction
//' @param lt_date numeric; truncation date as days since 1970-01-01
//' @param n_threads integer; OpenMP threads (0 = all available)
//' @return A list with elements `likelihoods` and (if grads) `gradients`.
//' @export
// [[Rcpp::export]]
 Rcpp::List screening_model_likes(
     int model,
     Rcpp::List inputs,
     Rcpp::List onset_pars,
     Rcpp::List sojourn_pars,
     Rcpp::List biom_pars,
     bool grads = false,
     double tol = 1e-6,
     std::string return_type = "",
     Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
     bool left_trunc = false,
     double lt_date = 0.0,
     int n_threads = 0)
 {
   // 1. Extract parameters from nested lists
   std::string onset_dist = get_s(onset_pars, "dist", "mvk");
   double o_shape = get_d(onset_pars, "shape", 1.0);
   double o_scale = get_d(onset_pars, "scale", 1.0);
   double o_A     = get_d(onset_pars, "A", -0.1);
   double o_B     = get_d(onset_pars, "B", 1e-4);
   double o_delta = get_d(onset_pars, "delta", 1e-4);
   double o_alpha = get_d(onset_pars, "alpha", 1e-4);
   double o_beta  = get_d(onset_pars, "beta", 0.1);
   double o_kappa = get_d(onset_pars, "kappa", 0.01);
   
   std::string sojourn_dist = get_s(sojourn_pars, "dist", "exp");
   double s_shape = get_d(sojourn_pars, "shape", 1.0);
   double s_scale = get_d(sojourn_pars, "scale", 1.0);
   double s_rate  = get_d(sojourn_pars, "rate", 0.1);
   double s_mulog = get_d(sojourn_pars, "mulog", 2.3);
   double s_sdlog = get_d(sojourn_pars, "sdlog", 0.6);
   
   double b_beta         = get_d(biom_pars, "beta", 0.05);
   double b_PrFalseNeg   = get_d(biom_pars, "PrFalseNeg", 0.05);
   double b_PrFalseNegBx = get_d(biom_pars, "PrFalseNegBx", 0.05);
   double b_beta0        = get_d(biom_pars, "beta0", -3.0);
   double b_beta1        = get_d(biom_pars, "beta1", 1.0);
   double b_b0_psa       = get_d(biom_pars, "b0_psa", -1.6094);
   double b_b1_psa       = get_d(biom_pars, "b1_psa", 0.0200);
   double b_b2_psa       = get_d(biom_pars, "b2_psa", 0.1094);
   double b_sigma_psa    = get_d(biom_pars, "sigma_psa", 0.2879);
   double b_mu_b0        = get_d(biom_pars, "mu_b0", -1.6094);
   double b_sigma_b0     = get_d(biom_pars, "sigma_b0", 0.2383);
   
   if (grads && sojourn_dist == "lognorm") {
     Rcpp::stop("AD gradients not implemented for lognormal sojourn yet.");
   }
   
   // 2. Standard Evaluation (No Gradients)
   if (!grads) {
     using density_f = std::function<double(double)>;
     using survival_f = std::function<double(double)>;
     
     density_f f1, f2; survival_f S1, S2;
     
     if (onset_dist == "weibull") {
       f1 = [=](double u){ return screening::dist::dweibull_t<double>(u, o_shape, o_scale); };
       S1 = [=](double u){ return screening::dist::pweibull_t<double>(u, o_shape, o_scale, false); };
     } else if (onset_dist == "beard") {
       f1 = [=](double u){ return screening::dist::dBeard_t<double>(u, o_alpha, o_beta, o_kappa); };
       S1 = [=](double u){ return screening::dist::pBeard_t<double>(u, o_alpha, o_beta, o_kappa, false); };
     } else {
       f1 = [=](double u){ return screening::dist::dMVK_t<double>(u, o_A, o_B, o_delta); };
       S1 = [=](double u){ return screening::dist::pMVK_t<double>(u, o_A, o_B, o_delta, false); };
     }
     
     if (sojourn_dist == "weibull") {
       f2 = [=](double u){ return screening::dist::dweibull_t<double>(u, s_shape, s_scale); };
       S2 = [=](double u){ return screening::dist::pweibull_t<double>(u, s_shape, s_scale, false); };
     } else if (sojourn_dist == "lognorm") {
       f2 = [=](double u){ return screening::dist::dlnorm_t(u, s_mulog, s_sdlog); };
       S2 = [=](double u){ return screening::dist::plnorm_t(u, s_mulog, s_sdlog, false); };
     } else {
       f2 = [=](double u){ return screening::dist::dexp_t<double>(u, s_rate); };
       S2 = [=](double u){ return screening::dist::pexp_t<double>(u, s_rate, false); };
     }
     
     std::vector<double> w;
     if (weights.isNotNull()) w = as<std::vector<double>>(weights);
     
     std::vector<double> likes_val;
     
     if (model == 1) {
       screening::ScreeningModel1<density_f, survival_f, density_f, survival_f, double> m(f1, S1, f2, S2, b_beta, tol);
       likes_val = m.likes(inputs);
     } else if (model == 2) {
       screening::ScreeningModel2<density_f, survival_f, density_f, survival_f, std::function<double(double)>, double> m(
           f1, S1, f2, S2, [=](double){ return b_PrFalseNeg; }, tol);
       likes_val = m.likes(inputs);
     } else if (model == 3) {
       screening::ScreeningModel3<density_f, survival_f, density_f, survival_f, std::function<double(double)>, double> m(
           f1, S1, f2, S2, [=](double y){ return 1.0/(1.0+std::exp(-(b_beta0+b_beta1*std::log(y)))); }, b_PrFalseNegBx, tol);
       likes_val = m.likes(inputs, 1e-12, return_type, w, left_trunc, R_NilValue);
     } else if (model == 4) {
       screening::ScreeningModel4<density_f, survival_f, density_f, survival_f,
                                  std::function<double(double)>,
                                  std::function<double(double, double, double)>, double> m(
                                      f1, S1, f2, S2,
                                      [=](double y) { return 1.0 / (1.0 + std::exp(-(b_beta0 + b_beta1 * std::log(y)))); },
                                      [=](double y, double age, double x) {
                                        if (y <= 0.0) return 0.0;
                                        double yrs = std::max(0.0, age - x);
                                        double mu = b_b0_psa + b_b1_psa * (age - 35.0) + b_b2_psa * yrs;
                                        double z = (std::log(y) - mu) / b_sigma_psa;
                                        return (1.0 / (y * b_sigma_psa * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
                                      }, b_PrFalseNegBx, tol);
       likes_val = m.likes(inputs, 1e-12, return_type, w, left_trunc, lt_date);
     } else if (model == 5) {
       if (!biom_pars.containsElementNamed("gh_nodes") || !biom_pars.containsElementNamed("gh_weights")) {
         Rcpp::stop("gh_nodes and gh_weights must be provided in biom_pars for Model 5.");
       }
       std::vector<double> nodes_vec = as<std::vector<double>>(biom_pars["gh_nodes"]);
       std::vector<double> weights_vec = as<std::vector<double>>(biom_pars["gh_weights"]);
       
       screening::ScreeningModel5<density_f, survival_f, density_f, survival_f,
                                  std::function<double(double)>, std::function<double(double, double, double, double)>, double> m(
                                      f1, S1, f2, S2,
                                      [=](double y) { return 1.0 / (1.0 + std::exp(-(b_beta0 + b_beta1 * std::log(y)))); },
                                      [=](double y, double age, double x, double b0) {
                                        if (y <= 0.0) return 0.0;
                                        double yrs = std::max(0.0, age - x);
                                        double mu = b0 + b_b1_psa * (age - 35.0) + b_b2_psa * yrs;
                                        double z = (std::log(y) - mu) / b_sigma_psa;
                                        return (1.0 / (y * b_sigma_psa * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
                                      }, b_PrFalseNegBx, b_mu_b0, b_sigma_b0, nodes_vec, weights_vec, tol);
       likes_val = m.likes(inputs, 1e-12, return_type, w, left_trunc, R_NilValue);
     } else {
       Rcpp::stop("Invalid model selection. Choose 1, 2, 3, 4, or 5.");
     }
     
     return Rcpp::List::create(Rcpp::Named("likelihoods") = likes_val);
   }
   
   // 3. Algorithmic Differentiation (Gradients)
   using cfaad::Number;
   size_t n_obs = inputs.size();
   std::vector<double> likes_val(n_obs);
   
   std::vector<std::string> param_names = {
     "o_shape", "o_scale", "o_A", "o_B", "o_delta",
     "o_alpha", "o_beta", "o_kappa", 
     "s_shape", "s_scale", "s_rate",
     "b_beta", "b_PrFalseNeg", "b_PrFalseNegBx", "b_beta0", "b_beta1",
     "b_b0_psa", "b_b1_psa", "b_b2_psa", "b_sigma_psa", "b_mu_b0", "b_sigma_b0"
   };
   NumericMatrix gradients(n_obs, param_names.size());
   colnames(gradients) = wrap(param_names);
   
   // --- Subject data now includes dob for left truncation ---
   struct SubjectData { double t; int type; double dob; std::vector<double> ti; std::vector<double> yi; std::vector<int> bxi; };
   std::vector<SubjectData> data(n_obs);
   
   for (size_t i = 0; i < n_obs; i++) {
     Rcpp::List input = inputs(i);
     data[i].t = as<double>(input("t"));
     data[i].type = as<int>(input("type"));
     data[i].dob = 0.0;
     if (input.containsElementNamed("dob")) data[i].dob = as<double>(input("dob"));
     if (input.containsElementNamed("ti")) data[i].ti = as<std::vector<double>>(input("ti"));
     if (input.containsElementNamed("yi")) data[i].yi = as<std::vector<double>>(input("yi"));
     if (input.containsElementNamed("bxi")) data[i].bxi = as<std::vector<int>>(input("bxi"));
   }
   
   std::vector<double> w;
   bool use_weights = weights.isNotNull() && return_type == "weighted_ll";
   if (use_weights) w = as<std::vector<double>>(weights);
   
#ifdef _OPENMP
   if (n_threads <= 0) n_threads = omp_get_max_threads();
#else
   n_threads = 1;
#endif
   
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
  thread_local cfaad::Tape local_tape;
  Number::tape = &local_tape;
  
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
  for (size_t i = 0; i < n_obs; ++i) {
    Number::tape->rewind();
    
    Number no_shape(o_shape), no_scale(o_scale), no_A(o_A), no_B(o_B), no_delta(o_delta);
    Number no_alpha(o_alpha), no_beta(o_beta), no_kappa(o_kappa);
    Number ns_shape(s_shape), ns_scale(s_scale), ns_rate(s_rate);
    Number nb_beta(b_beta), nb_PrFalseNeg(b_PrFalseNeg), nb_PrFalseNegBx(b_PrFalseNegBx);
    Number nb_beta0(b_beta0), nb_beta1(b_beta1);
    Number nb_b0_psa(b_b0_psa), nb_b1_psa(b_b1_psa), nb_b2_psa(b_b2_psa), nb_sigma_psa(b_sigma_psa);
    Number nb_mu_b0(b_mu_b0), nb_sigma_b0(b_sigma_b0);
    
    no_shape.putOnTape(); no_scale.putOnTape(); no_A.putOnTape(); no_B.putOnTape(); no_delta.putOnTape();
    no_alpha.putOnTape(); no_beta.putOnTape(); no_kappa.putOnTape();
    ns_shape.putOnTape(); ns_scale.putOnTape(); ns_rate.putOnTape();
    nb_beta.putOnTape(); nb_PrFalseNeg.putOnTape(); nb_PrFalseNegBx.putOnTape();
    nb_beta0.putOnTape(); nb_beta1.putOnTape();
    nb_b0_psa.putOnTape(); nb_b1_psa.putOnTape(); nb_b2_psa.putOnTape(); nb_sigma_psa.putOnTape();
    nb_mu_b0.putOnTape(); nb_sigma_b0.putOnTape();
    
    using density_ad_f = std::function<Number(double)>;
    using survival_ad_f = std::function<Number(double)>;
    
    density_ad_f f1, f2; survival_ad_f S1, S2;
    
    if (onset_dist == "weibull") {
      f1 = [&](double u){ return screening::dist::dweibull_t<Number>(u, no_shape, no_scale); };
      S1 = [&](double u){ return screening::dist::pweibull_t<Number>(u, no_shape, no_scale, false); };
    } else if (onset_dist == "beard") {
      f1 = [&](double u){ return screening::dist::dBeard_t<Number>(u, no_alpha, no_beta, no_kappa); };
      S1 = [&](double u){ return screening::dist::pBeard_t<Number>(u, no_alpha, no_beta, no_kappa, false); };
    } else {
      f1 = [&](double u){ return screening::dist::dMVK_t<Number>(u, no_A, no_B, no_delta); };
      S1 = [&](double u){ return screening::dist::pMVK_t<Number>(u, no_A, no_B, no_delta, false); };
    }
    if (sojourn_dist == "weibull") {
      f2 = [&](double u){ return screening::dist::dweibull_t<Number>(u, ns_shape, ns_scale); };
      S2 = [&](double u){ return screening::dist::pweibull_t<Number>(u, ns_shape, ns_scale, false); };
    } else {
      f2 = [&](double u){ return screening::dist::dexp_t<Number>(u, ns_rate); };
      S2 = [&](double u){ return screening::dist::pexp_t<Number>(u, ns_rate, false); };
    }
    
    Number res(0.0);
    Number denom(1.0);
    double age_lt = 0.0;
    if (left_trunc && lt_date != 0.0) {
      age_lt = (lt_date - data[i].dob) / 365.25;
    }
    
    if (model == 1) {
      screening::ScreeningModel1<density_ad_f, survival_ad_f, density_ad_f, survival_ad_f, Number> m(f1, S1, f2, S2, nb_beta, tol);
      m.update(data[i].ti);
      if (data[i].type == 1) res = m.X(data[i].t) + m.Y(data[i].t);
      else if (data[i].type == 2) res = m.Y(data[i].t - 1e-12) * (Number(1.0) - m.beta);
      else if (data[i].type == 3) res = m.I(data[i].t);
      else res = Number(-1.0);
      
      if (age_lt > 1e-6) denom = m.X(age_lt) + m.Y(age_lt);
    } 
    else if (model == 2) {
      screening::ScreeningModel2<density_ad_f, survival_ad_f, density_ad_f, survival_ad_f, std::function<Number(double)>, Number> m(
          f1, S1, f2, S2, [&](double){ return nb_PrFalseNeg; }, tol);
      m.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size());
      if (data[i].type == 1) res = m.X(data[i].t) + m.Y(data[i].t);
      else if (data[i].type == 2) res = (m.fulln > 0) ? m.Y(data[i].t - 1e-12) * (Number(1.0) - m.PrFalseNeg(m.yi[m.n-1])) : Number(0.0);
      else if (data[i].type == 3) res = m.I(data[i].t);
      else res = Number(-1.0);
      
      if (age_lt > 1e-6) denom = m.X(age_lt) + m.Y(age_lt);
    } 
    else if (model == 3) {
      screening::ScreeningModel3<density_ad_f, survival_ad_f, density_ad_f, survival_ad_f, std::function<Number(double)>, Number> m(
          f1, S1, f2, S2, [&](double y){ return Number(1.0)/(Number(1.0)+cfaad::exp(-(nb_beta0+nb_beta1*std::log(y)))); }, nb_PrFalseNegBx, tol);
      m.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size(), data[i].bxi.data(), data[i].bxi.size());
      if (data[i].type == 1) res = m.like_neg_screening(data[i].t);
      else if (data[i].type == 2) res = m.like_screen_detected_cancer(data[i].t);
      else if (data[i].type == 3) res = m.like_interval_cancer(data[i].t);
      else res = Number(-1.0);
      
      if (age_lt > 1e-6) denom = m.like_neg_screening(age_lt);
    } 
    else if (model == 4) {
      screening::ScreeningModel4<density_ad_f, survival_ad_f, density_ad_f, survival_ad_f,
                                 std::function<Number(double)>, std::function<Number(double, double, Number)>, Number> m(
                                     f1, S1, f2, S2,
                                     [&](double y) -> Number { return Number(1.0) / (Number(1.0) + cfaad::exp(-(nb_beta0 + nb_beta1 * std::log(y)))); },
                                     [&](double y, double age, Number x) -> Number {
                                       if (y <= 0.0) return Number(0.0);
                                       double yrs = std::max(0.0, age - screening::as_double(x));
                                       Number mu = nb_b0_psa + nb_b1_psa * (age - 35.0) + nb_b2_psa * yrs;
                                       Number z = (Number(std::log(y)) - mu) / nb_sigma_psa;
                                       return (Number(1.0) / (Number(y) * nb_sigma_psa * std::sqrt(2.0 * M_PI))) * cfaad::exp(Number(-0.5) * z * z);
                                     }, nb_PrFalseNegBx, tol);
      m.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size(), data[i].bxi.data(), data[i].bxi.size());
      if (data[i].type == 1) res = m.like_neg_screening(data[i].t);
      else if (data[i].type == 2) res = m.like_screen_detected_cancer(data[i].t);
      else if (data[i].type == 3) res = m.like_interval_cancer(data[i].t);
      else res = Number(-1.0);
      
      if (age_lt > 1e-6) denom = m.like_neg_screening(age_lt);
    } 
    else if (model == 5) {
      std::vector<double> nodes_vec = as<std::vector<double>>(biom_pars["gh_nodes"]);
      std::vector<double> weights_vec = as<std::vector<double>>(biom_pars["gh_weights"]);
      
      screening::ScreeningModel5<density_ad_f, survival_ad_f, density_ad_f, survival_ad_f,
                                 std::function<Number(double)>, std::function<Number(double, double, Number, Number)>, Number> m(
                                     f1, S1, f2, S2,
                                     [&](double y) -> Number { return Number(1.0) / (Number(1.0) + cfaad::exp(-(nb_beta0 + nb_beta1 * std::log(y)))); },
                                     [&](double y, double age, Number x, Number b0) -> Number {
                                       if (y <= 0.0) return Number(0.0);
                                       double yrs = std::max(0.0, age - screening::as_double(x));
                                       Number mu = b0 + nb_b1_psa * (age - 35.0) + nb_b2_psa * yrs;
                                       Number z = (Number(std::log(y)) - mu) / nb_sigma_psa;
                                       return (Number(1.0) / (Number(y) * nb_sigma_psa * std::sqrt(2.0 * M_PI))) * cfaad::exp(Number(-0.5) * z * z);
                                     }, nb_PrFalseNegBx, nb_mu_b0, nb_sigma_b0, nodes_vec, weights_vec, tol);
      m.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size(), data[i].bxi.data(), data[i].bxi.size());
      
      for(size_t k = 0; k < nodes_vec.size(); ++k) {
        Number b0_k = nb_mu_b0 + nb_sigma_b0 * Number(1.4142135623730951 * nodes_vec[k]);
        Number cond_L(0.0);
        if (data[i].type == 1) cond_L = m.like_neg_screening_cond(data[i].t, b0_k);
        else if (data[i].type == 2) cond_L = m.like_screen_detected_cancer_cond(data[i].t, b0_k);
        else if (data[i].type == 3) cond_L = m.like_interval_cancer_cond(data[i].t, b0_k);
        else cond_L = Number(-1.0);
        res += cond_L * Number(weights_vec[k] * 0.5641895835477563);
      }
      
      if (age_lt > 1e-6) {
        denom = Number(0.0);
        for(size_t k = 0; k < nodes_vec.size(); ++k) {
          Number b0_k = nb_mu_b0 + nb_sigma_b0 * Number(1.4142135623730951 * nodes_vec[k]);
          denom += m.like_neg_screening_cond(age_lt, b0_k) * Number(weights_vec[k] * 0.5641895835477563);
        }
      }
    }
    
    // ---------------------------------------------------------------
    // Left truncation adjustment (AD-aware: gradients flow through)
    // Simple LT at general lt_date: divide by P(History <= age_lt AND No event by age_lt)
    // ---------------------------------------------------------------
    if (left_trunc && lt_date != 0.0 && age_lt > 1e-6) {
      res = res / denom;
    }
    
    res.propagateToStart();
    double l_val = res.value();
    
    // Compute weighted log-likelihood multiplier if requested
    double grad_mult = 1.0;
    if (use_weights) {
      l_val = w[i] * std::log(l_val);
      grad_mult = w[i] / res.value(); // d/dx (w*log(L)) = w/L * dL/dx
    }
    likes_val[i] = l_val;
    
    gradients(i, 0)  = no_shape.adjoint() * grad_mult;
    gradients(i, 1)  = no_scale.adjoint() * grad_mult;
    gradients(i, 2)  = no_A.adjoint() * grad_mult;
    gradients(i, 3)  = no_B.adjoint() * grad_mult;
    gradients(i, 4)  = no_delta.adjoint() * grad_mult;
    gradients(i, 5)  = no_alpha.adjoint() * grad_mult;
    gradients(i, 6)  = no_beta.adjoint() * grad_mult;
    gradients(i, 7)  = no_kappa.adjoint() * grad_mult;
    gradients(i, 8)  = ns_shape.adjoint() * grad_mult;
    gradients(i, 9)  = ns_scale.adjoint() * grad_mult;
    gradients(i, 10) = ns_rate.adjoint() * grad_mult;
    gradients(i, 11) = nb_beta.adjoint() * grad_mult;
    gradients(i, 12) = nb_PrFalseNeg.adjoint() * grad_mult;
    gradients(i, 13) = nb_PrFalseNegBx.adjoint() * grad_mult;
    gradients(i, 14) = nb_beta0.adjoint() * grad_mult;
    gradients(i, 15) = nb_beta1.adjoint() * grad_mult;
    gradients(i, 16) = nb_b0_psa.adjoint() * grad_mult;
    gradients(i, 17) = nb_b1_psa.adjoint() * grad_mult;
    gradients(i, 18) = nb_b2_psa.adjoint() * grad_mult;
    gradients(i, 19) = nb_sigma_psa.adjoint() * grad_mult;
    gradients(i, 20) = nb_mu_b0.adjoint() * grad_mult;
    gradients(i, 21) = nb_sigma_b0.adjoint() * grad_mult;
  }
  Number::tape->clear();
}

return Rcpp::List::create(
  Rcpp::Named("likelihoods") = likes_val, 
  Rcpp::Named("gradients")   = gradients
);
 }

// =============================================================================
// Predictions: state-probability curves for a single screening schedule
// =============================================================================
//
// Internal helpers live in an anonymous namespace so they have internal
// linkage and do not pollute the translation unit's symbol table. They are
// small and single-purpose by design: each builds or validates exactly one
// piece of the public API's input. If screening_model_likes is ever
// refactored to share this logic, these helpers can be promoted.

namespace {

// Bundle of (density, survival) function objects for a distribution.
struct DistFns {
  std::function<double(double)> f;
  std::function<double(double)> S;
};
   
   // Build (f1, S1) for the onset distribution from onset_pars$dist.
   // Supported: "weibull", "beard", "mvk" (default).
   inline DistFns make_onset_d(const Rcpp::List& pars) {
     const std::string dist = get_s(pars, "dist", "mvk");
     DistFns out;
     if (dist == "weibull") {
       const double shape = get_d(pars, "shape", 1.0);
       const double scale = get_d(pars, "scale", 1.0);
       out.f = [=](double u) { return screening::dist::dweibull_t<double>(u, shape, scale); };
       out.S = [=](double u) { return screening::dist::pweibull_t<double>(u, shape, scale, false); };
     } else if (dist == "beard") {
       const double alpha = get_d(pars, "alpha", 1e-4);
       const double beta  = get_d(pars, "beta",  0.1);
       const double kappa = get_d(pars, "kappa", 0.01);
       out.f = [=](double u) { return screening::dist::dBeard_t<double>(u, alpha, beta, kappa); };
       out.S = [=](double u) { return screening::dist::pBeard_t<double>(u, alpha, beta, kappa, false); };
     } else if (dist == "mvk") {
       const double A     = get_d(pars, "A",     -0.1);
       const double B     = get_d(pars, "B",     1e-4);
       const double delta = get_d(pars, "delta", 1e-4);
       out.f = [=](double u) { return screening::dist::dMVK_t<double>(u, A, B, delta); };
       out.S = [=](double u) { return screening::dist::pMVK_t<double>(u, A, B, delta, false); };
     } else {
       Rcpp::stop("Unknown onset distribution '%s'. Supported: 'weibull', 'beard', 'mvk'.",
                  dist);
     }
     return out;
   }
   
   // Build (f2, S2) for the sojourn distribution from sojourn_pars$dist.
   // Supported: "weibull", "exp" (default), "lognorm".
   inline DistFns make_sojourn_d(const Rcpp::List& pars) {
     const std::string dist = get_s(pars, "dist", "exp");
     DistFns out;
     if (dist == "weibull") {
       const double shape = get_d(pars, "shape", 1.0);
       const double scale = get_d(pars, "scale", 1.0);
       out.f = [=](double u) { return screening::dist::dweibull_t<double>(u, shape, scale); };
       out.S = [=](double u) { return screening::dist::pweibull_t<double>(u, shape, scale, false); };
     } else if (dist == "lognorm") {
       const double mulog = get_d(pars, "mulog", 2.3);
       const double sdlog = get_d(pars, "sdlog", 0.6);
       out.f = [=](double u) { return screening::dist::dlnorm_t(u, mulog, sdlog); };
       out.S = [=](double u) { return screening::dist::plnorm_t(u, mulog, sdlog, false); };
     } else if (dist == "exp") {
       const double rate = get_d(pars, "rate", 0.1);
       out.f = [=](double u) { return screening::dist::dexp_t<double>(u, rate); };
       out.S = [=](double u) { return screening::dist::pexp_t<double>(u, rate, false); };
     } else {
       Rcpp::stop("Unknown sojourn distribution '%s'. Supported: 'weibull', 'exp', 'lognorm'.",
                  dist);
     }
     return out;
   }
   
   // Parsed screening history for a single subject.
   struct HistoryData {
     std::vector<double> ti;
     std::vector<double> yi;   // populated for models >= 2
     std::vector<int>    bxi;  // populated for models >= 3
   };
   
   // Parse and validate the history list against the required fields for `model`.
   // Throws (via Rcpp::stop) with an actionable message on any missing field or
   // length mismatch.
   inline HistoryData parse_history(const Rcpp::List& history, int model) {
     HistoryData h;
     if (!history.containsElementNamed("ti")) {
       Rcpp::stop("'history' must contain a numeric element 'ti'.");
     }
     h.ti = Rcpp::as<std::vector<double>>(history["ti"]);
     
     if (model >= 2) {
       if (!history.containsElementNamed("yi")) {
         Rcpp::stop("'history' must contain numeric element 'yi' for model %d.", model);
       }
       h.yi = Rcpp::as<std::vector<double>>(history["yi"]);
       if (h.yi.size() != h.ti.size()) {
         Rcpp::stop("Length of 'yi' (%d) must equal length of 'ti' (%d).",
                    (int) h.yi.size(), (int) h.ti.size());
       }
     }
     if (model >= 3) {
       if (!history.containsElementNamed("bxi")) {
         Rcpp::stop("'history' must contain integer element 'bxi' for model %d.", model);
       }
       h.bxi = Rcpp::as<std::vector<int>>(history["bxi"]);
       if (h.bxi.size() != h.ti.size()) {
         Rcpp::stop("Length of 'bxi' (%d) must equal length of 'ti' (%d).",
                    (int) h.bxi.size(), (int) h.ti.size());
       }
     }
     return h;
   }
   
 } // anonymous namespace

//' State probability predictions for a screening model
 //'
 //' @description
 //' Evaluates the marginal state probabilities \eqn{X} (healthy),
 //' \eqn{Y} (preclinical), \eqn{Z} (diagnosed) and the instantaneous incidence
 //' \eqn{I} across a vector of times for a single subject's screening
 //' schedule. Useful for visualising state trajectories under different
 //' screening regimes.
 //'
 //' @param model Integer identifying the screening model. Currently
 //'   supported: \code{1}, \code{2}, \code{3}. Prediction is not supported
 //'   for models 4 and 5; see Details.
 //' @param t Numeric vector of evaluation times (e.g. ages). Must be
 //'   non-empty.
 //' @param history Named list describing the screening history. Must
 //'   contain \code{ti} (numeric screening times). For \code{model = 2} must
 //'   also contain \code{yi} (biomarker values). For \code{model = 3} must
 //'   also contain \code{yi} and \code{bxi} (integer 0/1 biopsy indicators).
 //'   When \code{yi} and \code{bxi} are required they must have the same
 //'   length as \code{ti}.
 //' @param onset_pars Named list of onset distribution parameters. Choose
 //'   \code{dist} from \code{"weibull"}, \code{"beard"} or \code{"mvk"}
 //'   (default). See \code{\link{screening_model_likes}} for the parameter
 //'   names of each distribution.
 //' @param sojourn_pars Named list of sojourn distribution parameters.
 //'   Choose \code{dist} from \code{"weibull"}, \code{"exp"} (default) or
 //'   \code{"lognorm"}.
 //' @param biom_pars Named list of test-characteristic parameters:
 //'   \itemize{
 //'     \item Model 1: \code{beta} (constant false-negative fraction).
 //'     \item Model 2: \code{beta0}, \code{beta1} (logistic model for the
 //'       false-negative probability as a function of \code{log(yi)}).
 //'     \item Model 3: \code{beta0}, \code{beta1}, \code{PrFalseNegBx}
 //'       (false-negative biopsy probability given biopsy and cancer).
 //'   }
 //' @param simple Logical; if \code{TRUE} (default) \code{Z} is computed
 //'   as the complement \code{1 - (X + Y)}, which is fast. If \code{FALSE},
 //'   \code{Z} is computed by direct integration — slower, but useful as a
 //'   numerical sanity check.
 //' @param tol Positive numerical tolerance for the Gauss-Kronrod
 //'   integrations. Default \code{1e-6}.
 //'
 //' @details
 //' Models 4 and 5 route their screening-history contribution through a
 //' biomarker density and (for model 5) a random intercept integrated by
 //' Gauss-Hermite quadrature. The marginal state probabilities returned by
 //' this function are not meaningful for those models; use
 //' \code{\link{screening_model_likes}} instead.
 //'
 //' @return A \code{data.frame} with columns \code{t}, \code{X}, \code{Y},
 //'   \code{Z} and \code{I}, each of length \code{length(t)}.
 //'
 //' @seealso \code{\link{screening_model_likes}}
 //'
 //' @examples
 //' t <- seq(0, 30, length.out = 101)
 //'
 //' # No screening baseline
 //' res0 <- screening_model_predictions(
 //'     model = 1, t = t,
 //'     history = list(ti = numeric(0)),
 //'     onset_pars = list(dist = "weibull", shape = 1.5, scale = 20),
 //'     sojourn_pars = list(dist = "weibull", shape = 1.5, scale = 10),
 //'     biom_pars = list(beta = 0.4)
 //' )
 //'
 //' # Four screens between ages 10 and 25
 //' res <- screening_model_predictions(
 //'     model = 1, t = t,
 //'     history = list(ti = c(10, 15, 20, 25)),
 //'     onset_pars = list(dist = "weibull", shape = 1.5, scale = 20),
 //'     sojourn_pars = list(dist = "weibull", shape = 1.5, scale = 10),
 //'     biom_pars = list(beta = 0.4)
 //' )
 //' head(res)
 //'
 //' \donttest{
 //' op <- par(mfrow = c(1, 2))
 //' matplot(res0$t, as.matrix(res0[, c("X", "Y", "Z")]),
 //'         type = "l", lty = 1, col = 1:3,
 //'         xlab = "age", ylab = "probability", main = "No screening")
 //' matplot(res$t, as.matrix(res[, c("X", "Y", "Z")]),
 //'         type = "l", lty = 1, col = 1:3,
 //'         xlab = "age", ylab = "probability", main = "Screening")
 //' legend("right", legend = c("X", "Y", "Z"), col = 1:3, lty = 1, bty = "n")
 //' par(op)
 //' }
 //'
 //' @name screening_model_predictions
 //' @export
 // [[Rcpp::export]]
 Rcpp::DataFrame screening_model_predictions(
     int model,
     std::vector<double> t,
     Rcpp::List history,
     Rcpp::List onset_pars,
     Rcpp::List sojourn_pars,
     Rcpp::List biom_pars,
     bool simple = true,
     double tol = 1e-6)
 {
   // ---- input validation -----------------------------------------------------
   if (model < 1 || model > 5) {
     Rcpp::stop("'model' must be an integer between 1 and 5.");
   }
   if (model == 4 || model == 5) {
     Rcpp::stop(
       "Predictions are not supported for model %d; "
       "use model 1, 2 or 3, or call screening_model_likes().",
       model);
   }
   if (t.empty()) {
     Rcpp::stop("'t' must be a non-empty numeric vector.");
   }
   if (tol <= 0.0) {
     Rcpp::stop("'tol' must be positive (got %g).", tol);
   }
   
   const HistoryData h      = parse_history(history, model);
   const DistFns     onset  = make_onset_d(onset_pars);
   const DistFns     sojn   = make_sojourn_d(sojourn_pars);
   
   using density_f  = std::function<double(double)>;
   using survival_f = std::function<double(double)>;
   
   // ---- dispatch -------------------------------------------------------------
   if (model == 1) {
     const double beta = get_d(biom_pars, "beta", 0.05);
     screening::ScreeningModel1<density_f, survival_f, density_f, survival_f, double>
       m(onset.f, onset.S, sojn.f, sojn.S, beta, tol);
     m.update(h.ti);
     return m.predictions(t, simple);
   }
   
   if (model == 2) {
     const double beta0 = get_d(biom_pars, "beta0", -3.0);
     const double beta1 = get_d(biom_pars, "beta1",  1.0);
     auto pr_false_neg = [=](double y) {
       return 1.0 / (1.0 + std::exp(-(beta0 + beta1 * std::log(y))));
     };
     screening::ScreeningModel2<density_f, survival_f, density_f, survival_f,
                                std::function<double(double)>, double>
       m(onset.f, onset.S, sojn.f, sojn.S, pr_false_neg, tol);
     m.update(h.ti, h.yi);
     return m.predictions(t, simple);
   }
   
   // model == 3
   const double beta0        = get_d(biom_pars, "beta0",        -3.0);
   const double beta1        = get_d(biom_pars, "beta1",         1.0);
   const double PrFalseNegBx = get_d(biom_pars, "PrFalseNegBx",  0.05);
   auto pr_no_bx = [=](double y) {
     return 1.0 / (1.0 + std::exp(-(beta0 + beta1 * std::log(y))));
   };
   screening::ScreeningModel3<density_f, survival_f, density_f, survival_f,
                              std::function<double(double)>, double>
     m(onset.f, onset.S, sojn.f, sojn.S, pr_no_bx, PrFalseNegBx, tol);
   m.update(h.ti, h.yi, h.bxi);
   return m.predictions(t, simple);
 }