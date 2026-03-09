#include <Rcpp.h>
#include "screening.h"

#define _USE_MATH_DEFINES // for pi
#include <cmath>
#include <array>
#include <limits>
#include <iostream>
#include <vector>

#include "cfaad/AAD.h"
#include "cfaad/AADInit.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------
// AD Compiler Helpers for cfaad and Boost
// -------------------------------------------------------------
namespace cfaad {
// Teach C++ streams how to print a cfaad::Number (required by Boost Error Handling)
inline std::ostream& operator<<(std::ostream& os, const Number& n) {
os << n.value();
return os;
}

// Teach C++ how to take the absolute value of cfaad types (required by Boost Quadrature)
inline Number abs(const Number& n) {
return n.value() < 0.0 ? Number(-n) : n;
}
template <typename L, typename R, typename OP>
inline Number abs(const BinaryExpression<L, R, OP>& expr) {
Number n(expr);
return n.value() < 0.0 ? Number(-n) : n;
}
template <typename E, typename OP>
inline Number abs(const UnaryExpression<E, OP>& expr) {
Number n(expr);
return n.value() < 0.0 ? Number(-n) : n;
}
}

// Boost Quadrature needs numeric limits to compile generic AD types safely.
namespace std {
template <>
struct numeric_limits<cfaad::Number> : public numeric_limits<double> {
static cfaad::Number min() { return cfaad::Number(numeric_limits<double>::min()); }
static cfaad::Number max() { return cfaad::Number(numeric_limits<double>::max()); }
static cfaad::Number lowest() { return cfaad::Number(numeric_limits<double>::lowest()); }
static cfaad::Number epsilon() { return cfaad::Number(numeric_limits<double>::epsilon()); }
static cfaad::Number round_error() { return cfaad::Number(numeric_limits<double>::round_error()); }
static cfaad::Number infinity() { return cfaad::Number(numeric_limits<double>::infinity()); }
static cfaad::Number quiet_NaN() { return cfaad::Number(numeric_limits<double>::quiet_NaN()); }
static cfaad::Number signaling_NaN() { return cfaad::Number(numeric_limits<double>::signaling_NaN()); }
static cfaad::Number denorm_min() { return cfaad::Number(numeric_limits<double>::denorm_min()); }
};
}

// -------------------------------------------------------------
// Original Standard Functions
// -------------------------------------------------------------

//' Do predictions for ScreeningModel1
//' @name ScreeningModel1
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_1_predictions(t=x, tj=vector("double"), shape1=1.5, shape2=1.5, scale1=20, scale2=10, beta=0.4) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_1_predictions(t=x, tj=c(10,15, 20, 25), shape1=1.5, shape2=1.5, scale1=20, scale2=10, beta=0.4) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_1_predictions(std::vector<double> t,
                            std::vector<double> ti,
                            double shape1=1,
                            double scale1=1,
                            double shape2=1,
                            double scale2=1,
                            double beta=0.05,
                            bool simple = true,
                            double tol = 1e-6) {
// Initialize the ScreeningModel with appropriate parameters
screening::ScreeningModel1 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
                             [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
                             [&](double u) { return R::dweibull(u,shape2,scale2,0); },
                             [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
                             beta,
                             tol);
m.update(ti);
return m.predictions(t,simple);
}

//' Do likelihood calculations for ScreeningModel1
//' @name ScreeningModel1
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_1_likes(Rcpp::List inputs, double shape1 = 1.0, double scale1 = 1.0, 
                        double shape2 = 1.0, double scale2 = 1.0, double beta = 0.05,
                        double tol=1e-6) {
  screening::ScreeningModel1 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
                               [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
                               [&](double u) { return R::dweibull(u,shape2,scale2,0); },
                               [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
                               beta,
                               tol);
  return m.likes(inputs);
}

//' Do predictions for ScreeningModel2
//' @name ScreeningModel2
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param yi double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_2_predictions(t=x, ti=vector("double"), yi=vector("double"), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_2_predictions(t=x, ti=c(10,15, 20, 25), yi=c(3,3,3), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_2_predictions(std::vector<double> t,
                            std::vector<double> ti,
                            std::vector<double> yi,
                            double shape1=1, double scale1=1,
                            double shape2=1, double scale2=1,
                            double beta0=-3.0,
                            double beta1=1.0,
                            bool simple = true,
                            double tol = 1e-6) {
// Initialize the model with appropriate parameters
screening::ScreeningModel2 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
                             [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
                             [&](double u) { return R::dweibull(u,shape2,scale2,0); },
                             [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
                             [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
                             tol);
m.update(ti,yi);
return m.predictions(t,simple);
}

//' Do likelihood calculations for ScreeningModel2
//' @name ScreeningModel2
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_2_likes(Rcpp::List inputs,
                        double shape1 = 1.0,
                        double scale1 = 1.0, 
                        double shape2 = 1.0,
                        double scale2 = 1.0, 
                        double beta0=-3.0,
                        double beta1=1.0,
                        double tol=1e-6) {
  screening::ScreeningModel2 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
                               [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
                               [&](double u) { return R::dweibull(u,shape2,scale2,0); },
                               [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
                               [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
                               tol);
  return m.likes(inputs);
}

//' Do predictions for ScreeningModel3
//' @name ScreeningModel3
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param yi double vector of screening times
//' @param bxi integer vector of biopsy choices
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_3_predictions(t=x, ti=vector("double"), yi=vector("double"), bxi=vector("integer"), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_3_predictions(t=x, ti=c(10,15, 20, 25), yi=c(3,3,3), bxi=c(FALSE,FALSE,TRUE), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_3_predictions(std::vector<double> t,
                            std::vector<double> ti,
                            std::vector<double> yi,
                            std::vector<int> bxi,
                            double shape1=1, double scale1=1,
                            double shape2=1, double scale2=1,
                            double beta0=-3.0,
                            double beta1=1.0,
                            double PrFalseNegBx=0.05,
                            bool simple = true,
                            double tol = 1e-6) {
// Initialize the model with appropriate parameters
screening::ScreeningModel3 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
                             [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
                             [&](double u) { return R::dweibull(u,shape2,scale2,0); },
                             [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
                             [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
                             PrFalseNegBx,
                             tol);
m.update(ti,yi,bxi);
return m.predictions(t,simple);
}

// these will end up inside openmp loop; same parametrisation like in R
inline double dweibull(double x, double shape, double scale) {
if (x < 0) return 0.0;
double xl = x / scale;
return (shape / scale) * std::pow(xl, shape - 1.0) * std::exp(-std::pow(xl, shape));
}
inline double pweibull(double x, double shape, double scale, bool lower_tail = false) {
if (x < 0) return lower_tail ? 0.0 : 1.0;
return lower_tail ? 1-std::exp(-std::pow(x / scale, shape)) : std::exp(-std::pow(x / scale, shape));
}

//' Do likelihood calculations for ScreeningModel3
//' @name ScreeningModel3
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape1 Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param return_type string, if "weighted_ll" returns sum of weighted log-likelihoods (default "")
//' @param weights vector of weights corresponding to inputs, required if return_type is "weighted_ll"
//' @return vector of likelihoods (or vector of length 1 containing weighted log-likelihood sum)
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_3_likes(Rcpp::List inputs,
                        double shape1 = 1.0,
                        double scale1 = 1.0, 
                        double shape2 = 1.0,
                        double scale2 = 1.0, 
                        double beta0=-3.0,
                        double beta1=1.0,
                        double PrFalseNegBx=0.05,
                        double tol=1e-6,
                        std::string return_type = "", 
                        Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue) {
  
  std::vector<double> w;
  
  if (weights.isNotNull()) {
    Rcpp::NumericVector weights_nv(weights);
    w = Rcpp::as<std::vector<double>>(weights_nv);
  }
  
  screening::ScreeningModel3 m([&](double u) { return dweibull(u,shape1,scale1); },
                               [&](double u) { return pweibull(u,shape1,scale1, 0); },
                               [&](double u) { return dweibull(u,shape2,scale2); },
                               [&](double u) { return pweibull(u,shape2,scale2, 0); },
                               [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
                               PrFalseNegBx,
                               tol);
  return m.likes(inputs, 1e-12, return_type, w);
}

// MVK distribution
// Reminder:
// nu: initiation rate
// alpha: cell division rate
// beta: cell death rate
// mu: malignant transformation rate
// A=(beta+mu-alpha - sqrt((beta+mu-alpha)^2+4*mu*alpha))/2
// B=(beta+mu-alpha + sqrt((beta+mu-alpha)^2+4*mu*alpha))/2
// g=-(A+B)=alpha-beta-mu is approximately equal to the net proliferation rate
// B: upper bound for the malignant transformation rate
// delta=nu/alpha
// B-A = sqrt((beta+mu-alpha)^2+4*mu*alpha)
//
// Bounds:
// (nu, alpha, beta, mu) >= 0
// delta>0, B>=0, A<=0 => B-A>=B>=0
inline double dMVK(double t, double A, double B, double delta) {
double P = std::expm1((B-A) * t) * std::exp(B * delta * t) * std::pow(B - A, delta);
double Q = std::pow(B * std::exp((B - A) * t) - A, 1+delta);
double val = - delta * A * B * P/Q;
if (!std::isfinite(val)) val = 0.0;
return val;
}
inline double pMVK(double t, double A, double B, double delta, bool lower_tail = true) {
double logS = delta*(std::log(B-A) + B*t - std::log(B*std::exp((B-A)*t) - A));
return lower_tail ? -std::expm1(logS) : std::exp(logS);
}

//' Do likelihood calculations for ScreeningModel3 using MVK onset
//' @name ScreeningModel3MVK
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param shape2 Weibull shape for clinical diagnosis (Sojourn density)
//' @param scale2 Weibull scale for clinical diagnosis (Sojourn density)
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param return_type string, if "weighted_ll" returns sum of weighted log-likelihoods (default "")
//' @param weights vector of weights corresponding to inputs, required if return_type is "weighted_ll"
//' @param left_trunc bool, apply left truncation adjustment (default false)
//' @param incidence DataFrame containing background incidence rates, required if left_trunc is true
//' @return vector of likelihoods (or vector of length 1 containing weighted log-likelihood sum)
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_3_likes_MVK(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double shape2 = 1,
   double scale2 = 1,
   double beta0=-3.0,
   double beta1=1.0,
   double PrFalseNegBx=0.05,
   double tol = 1e-6,
   std::string return_type = "",
   Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
   bool left_trunc = false,
   Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue){
 
 std::vector<double> w;
 
 if(weights.isNotNull()) {
   Rcpp::NumericVector weights_nv(weights);
   w = Rcpp::as<std::vector<double>>(weights_nv);
 }
 
 screening::ScreeningModel3 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dweibull(u,shape2,scale2); },
                              [&](double u){ return pweibull(u,shape2,scale2, 0); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              PrFalseNegBx,
                              tol);
 
 return m.likes(inputs, 1e-12, return_type, w, left_trunc, incidence);
}

inline double dlnorm(double t, double mulog, double sdlog) {
return (t <= 0) ? 0.0 : (1.0 / (t * sdlog * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * std::pow((std::log(t) - mulog) / sdlog, 2.0));
}

inline double plnorm(double t, double mulog, double sdlog, bool lower_tail = true) {
return (t <= 0) ? (lower_tail ? 0.0 : 1.0) : 0.5 * std::erfc(((lower_tail ? -1.0 : 1.0) * (std::log(t) - mulog)) / (sdlog * std::sqrt(2.0)));
}

//' Do likelihood calculations for ScreeningModel3 using MVK onset and Lognormal Sojourn
//' @name ScreeningModel3MVKLognorm
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param mulog Lognormal meanlog for clinical diagnosis (Sojourn density, default=2.3 ~ 10 yrs)
//' @param sdlog Lognormal sdlog for clinical diagnosis (Sojourn density, default=0.6)
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param return_type string, if "weighted_ll" returns sum of weighted log-likelihoods (default "")
//' @param weights vector of weights corresponding to inputs, required if return_type is "weighted_ll"
//' @param left_trunc bool, apply left truncation adjustment (default false)
//' @param incidence DataFrame containing background incidence rates, required if left_trunc is true
//' @return vector of likelihoods (or vector of length 1 containing weighted log-likelihood sum)
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_3_likes_MVK_lognorm(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double mulog = 2.3, 
   double sdlog = 0.6,
   double beta0 = -3.0,
   double beta1 = 1.0,
   double PrFalseNegBx = 0.05,
   double tol = 1e-6,
   std::string return_type = "",
   Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
   bool left_trunc = false,
   Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
 
 std::vector<double> w;
 
 if(weights.isNotNull()) {
   Rcpp::NumericVector weights_nv(weights);
   w = Rcpp::as<std::vector<double>>(weights_nv);
 }
 
 screening::ScreeningModel3 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dlnorm(u, mulog, sdlog); },
                              [&](double u){ return plnorm(u, mulog, sdlog, false); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              PrFalseNegBx,
                              tol);
 
 return m.likes(inputs, 1e-12, return_type, w, left_trunc, incidence);
}

inline double dexp(double t, double rate) {
return (t < 0) ? 0.0 : rate * std::exp(-rate * t);
}

inline double pexp(double t, double rate, bool lower_tail = true) {
if (t < 0) return lower_tail ? 0.0 : 1.0;
return lower_tail ? 1.0 - std::exp(-rate * t) : std::exp(-rate * t);
}

//' Do likelihood calculations for ScreeningModel3 using MVK onset and Exponential Sojourn
//' @name ScreeningModel3MVKExp
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param rate Exponential rate for clinical diagnosis (Sojourn density, default=0.1 ~ mean 10 yrs)
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param return_type string, if "weighted_ll" returns sum of weighted log-likelihoods (default "")
//' @param weights vector of weights corresponding to inputs, required if return_type is "weighted_ll"
//' @param left_trunc bool, apply left truncation adjustment (default false)
//' @param incidence DataFrame containing background incidence rates, required if left_trunc is true
//' @return vector of likelihoods (or vector of length 1 containing weighted log-likelihood sum)
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_3_likes_MVK_exp(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double rate = 0.1,
   double beta0 = -3.0,
   double beta1 = 1.0,
   double PrFalseNegBx = 0.05,
   double tol = 1e-6,
   std::string return_type = "",
   Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
   bool left_trunc = false,
   Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
 
 std::vector<double> w;
 
 if(weights.isNotNull()) {
   Rcpp::NumericVector weights_nv(weights);
   w = Rcpp::as<std::vector<double>>(weights_nv);
 }
 
 screening::ScreeningModel3 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dexp(u, rate); },
                              [&](double u){ return pexp(u, rate, false); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              PrFalseNegBx,
                              tol);
 
 return m.likes(inputs, 1e-12, return_type, w, left_trunc, incidence);
}

//' Do likelihood calculations for ScreeningModel2 using MVK onset and Exponential Sojourn
//' @name ScreeningModel2MVKExp
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param rate Exponential rate for clinical diagnosis (Sojourn density, default=0.1 ~ mean 10 yrs)
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_2_likes_MVK_exp(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double rate = 0.1,
   double beta0 = -3.0,
   double beta1 = 1.0,
   double tol = 1e-6) {
 
 screening::ScreeningModel2 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dexp(u, rate); },
                              [&](double u){ return pexp(u, rate, false); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              tol);
 
 return m.likes(inputs);
}

//' Do likelihood calculations for ScreeningModel1 using MVK onset and Exponential Sojourn
//' @name ScreeningModel1MVKExp
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param rate Exponential rate for clinical diagnosis (Sojourn density, default=0.1 ~ mean 10 yrs)
//' @param beta false negative fraction for screening
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_1_likes_MVK_exp(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double rate = 0.1,
   double beta = 0.05,
   double tol = 1e-6) {
 
 screening::ScreeningModel1<
   std::function<double(double)>, std::function<double(double)>, 
   std::function<double(double)>, std::function<double(double)>, double> m(
       [&](double u){ return dMVK(u, A, B, delta);},
       [&](double u){ return pMVK(u, A, B, delta, 0);},
       [&](double u){ return dexp(u, rate); },
       [&](double u){ return pexp(u, rate, false); },
       beta,
       tol);
 
 return m.likes(inputs);
}

//' Do likelihood calculations for ScreeningModel4 using MVK onset, Exponential Sojourn, and Log-linear PSA with change of slope for onset
//' @name ScreeningModel4LikesLoglin
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times, yi for biomarker values, bxi for biopsy indicators, and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (typically negative, related to net proliferation)
//' @param B MVK parameter B (typically positive, related to malignant transformation)
//' @param delta MVK parameter delta (ratio of initiation rate to cell division rate)
//' @param rate Exponential rate for clinical diagnosis (Sojourn density, default=0.1 ~ mean 10 yrs)
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param b0_psa intercept for log-linear PSA model
//' @param b1_psa slope for age for log-linear PSA model
//' @param b2_psa slope increment after onset for log-linear PSA model
//' @param sigma_psa standard deviation of log(PSA)
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param return_type string, if "weighted_ll" returns sum of weighted log-likelihoods (default "")
//' @param weights vector of weights corresponding to inputs, required if return_type is "weighted_ll"
//' @param left_trunc bool, apply left truncation adjustment (default false)
//' @param incidence DataFrame containing background incidence rates, required if left_trunc is true
//' @return vector of likelihoods (or vector of length 1 containing weighted log-likelihood sum)
//' @export
// [[Rcpp::export]]
std::vector<double> screening_model_4_likes_loglin(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double rate = 0.1,
   double beta0 = 3.2892,
   double beta1 = -0.5533,
   double b0_psa = -1.6094,
   double b1_psa = 0.0200,
   double b2_psa = 0.1094,
   double sigma_psa = 0.2879,
   double PrFalseNegBx = 0.0,
   double tol = 1e-6,
   std::string return_type = "",
   Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
   bool left_trunc = false,
   Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
 
 std::vector<double> w;
 if(weights.isNotNull()) {
   Rcpp::NumericVector weights_nv(weights);
   w = Rcpp::as<std::vector<double>>(weights_nv);
 }
 
 screening::ScreeningModel4 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dexp(u, rate); },
                              [&](double u){ return pexp(u, rate, false); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              [&](double y, double age, double x){ 
                                // x is the onset time. std::max handles x=1e9 in no_onset_history *= biomarker_den(yi[k], this->tj[k+1], 1e9); 
                                double mu = b0_psa + b1_psa * (age - 35.0) + b2_psa * std::max(0.0, age - x);
                                if (y <= 0) return 0.0;
                                return (1.0 / (y * sigma_psa * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * std::pow((std::log(y) - mu) / sigma_psa, 2.0));
                              },
                              PrFalseNegBx,
                              tol);
 
 return m.likes(inputs, 1e-12, return_type, w, left_trunc, incidence);
}

// -------------------------------------------------------------
// Algorithmic Differentiation Additions
// -------------------------------------------------------------

template <typename T>
inline T dMVK_t(double t, T A, T B, T delta) {
// Using exp(x) - 1.0 as cfaad does not provide expm1
T P = (cfaad::exp((B - A) * t) - T(1.0)) * cfaad::exp(B * delta * t) * cfaad::pow(B - A, delta);
T Q = cfaad::pow(B * cfaad::exp((B - A) * t) - A, T(1.0) + delta);
return -delta * A * B * P / Q;
}

template <typename T>
inline T pMVK_t(double t, T A, T B, T delta, bool lower_tail = true) {
T logS = delta * (cfaad::log(B - A) + B * t - cfaad::log(B * cfaad::exp((B - A) * t) - A));
if (lower_tail) {
  return T(1.0) - cfaad::exp(logS);
} else {
  return cfaad::exp(logS);
}
}

template <typename T>
inline T dexp_t(double t, T rate) {
return (t < 0) ? T(0.0) : rate * cfaad::exp(-rate * t);
}

template <typename T>
inline T pexp_t(double t, T rate, bool lower_tail = true) {
if (t < 0) {
  if (lower_tail) return T(0.0);
  else return T(1.0);
}

if (lower_tail) {
  return T(1.0) - cfaad::exp(-rate * t);
} else {
  return cfaad::exp(-rate * t);
}
}

//' Do AD likelihood and gradient calculations for ScreeningModel1 (MVK + Exp)
//' @name screening_model_1_likes_MVK_exp_grad
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param A MVK parameter A (active for AD)
//' @param B MVK parameter B (active for AD)
//' @param delta MVK parameter delta (active for AD)
//' @param rate Exponential rate for clinical diagnosis (active for AD)
//' @param beta false negative fraction for screening (active for AD)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param n_threads number of threads to use (default=0, auto-detects)
//' @return list containing likelihoods and gradients
//' @export
// [[Rcpp::export]]
Rcpp::List screening_model_1_likes_MVK_exp_grad(
   Rcpp::List inputs, double A = -0.1, double B = 1e-4, 
   double delta = 1e-4, double rate = 0.1, double beta = 0.05,
   double tol = 1e-6, int n_threads = 0) {
 
 using cfaad::Number;
 
 size_t n_obs = inputs.size();
 std::vector<double> likes_val(n_obs);
 Rcpp::NumericMatrix gradients(n_obs, 5);
 
 // Extract C++ types BEFORE OpenMP block to avoid R API multi-threading issues
 struct SubjectData {
   double t;
   int type;
   std::vector<double> ti;
 };
 
 std::vector<SubjectData> data(n_obs);
 for (size_t i = 0; i < n_obs; i++) {
   Rcpp::List input = inputs(i);
   data[i].t = Rcpp::as<double>(input("t"));
   data[i].type = Rcpp::as<int>(input("type"));
   data[i].ti = Rcpp::as<std::vector<double>>(input("ti"));
 }
 
 double eps = 1.0e-12;
 
#ifdef _OPENMP
 if (n_threads <= 0) {
   n_threads = omp_get_max_threads();
 }
#else
 n_threads = 1;
#endif
 
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
// Thread local tape prevents data races and tape size explosion
cfaad::Tape local_tape;
Number::tape = &local_tape;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
for (size_t i = 0; i < n_obs; ++i) {
  
  // REWIND: Optimal memory management (clears nodes but keeps block allocation)
  Number::tape->rewind(); 
  
  Number A_n(A);         A_n.putOnTape();
  Number B_n(B);         B_n.putOnTape();
  Number delta_n(delta); delta_n.putOnTape();
  Number rate_n(rate);   rate_n.putOnTape();
  Number beta_n(beta);   beta_n.putOnTape();
  
  screening::ScreeningModel1<
    std::function<Number(double)>, std::function<Number(double)>, 
    std::function<Number(double)>, std::function<Number(double)>, Number> 
    m(
      [&](double u) { return dMVK_t<Number>(u, A_n, B_n, delta_n); },
      [&](double u) { return pMVK_t<Number>(u, A_n, B_n, delta_n, false); },
      [&](double u) { return dexp_t<Number>(u, rate_n); },
      [&](double u) { return pexp_t<Number>(u, rate_n, false); },
      beta_n, tol
    );
  
  m.update(data[i].ti);
  
  Number res;
  if (data[i].type == 1) {  
    res = m.X(data[i].t) + m.Y(data[i].t);
  } else if (data[i].type == 2) {  
    res = m.Y(data[i].t - eps) * (Number(1.0) - m.beta);
  } else if (data[i].type == 3) {  
    res = m.I(data[i].t);
  } else {
    res = Number(-1.0);  
  }
  
  // Backpropagate exactly one subject at a time
  res.propagateToStart();
  
  likes_val[i] = res.value();
  gradients(i, 0) = A_n.adjoint();
  gradients(i, 1) = B_n.adjoint();
  gradients(i, 2) = delta_n.adjoint();
  gradients(i, 3) = rate_n.adjoint();
  gradients(i, 4) = beta_n.adjoint();
}

// Clear out memory before the thread exits
Number::tape->clear();
}

Rcpp::colnames(gradients) = Rcpp::CharacterVector::create("A", "B", "delta", "rate", "beta");

return Rcpp::List::create(
Rcpp::Named("likelihoods") = likes_val,
Rcpp::Named("gradients") = gradients
);
}

//' Do AD likelihood and gradient calculations for ScreeningModel4
//' (MVK onset + Exp sojourn + log-linear PSA with slope change after onset)
//' @name screening_model_4_likes_loglin_grad
//' @param inputs list of list with elements of t for the evaluation time,
//' ti for screening times, yi for biomarker values, bxi for biopsy indicators,
//' and type for the type of likelihood
//' (1 = no cancer detected, 2 = screen-detected cancer, 3 = interval cancer)
//' @param A MVK parameter A (active for AD)
//' @param B MVK parameter B (active for AD)
//' @param delta MVK parameter delta (active for AD)
//' @param rate Exponential rate for clinical diagnosis (active for AD)
//' @param beta0 intercept for logistic no-biopsy model (active for AD)
//' @param beta1 slope of log(yi) for logistic no-biopsy model (active for AD)
//' @param b0_psa intercept for log-linear PSA model (active for AD)
//' @param b1_psa age slope for log-linear PSA model (active for AD)
//' @param b2_psa slope increment after onset for log-linear PSA model (active for AD)
//' @param sigma_psa standard deviation of log(PSA) (active for AD)
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken (active for AD)
//' @param tol double for numeric integration tolerance
//' @param n_threads number of threads to use (default = 0, auto-detects)
//' @return list containing likelihoods and gradients
//' @export
// [[Rcpp::export]]
Rcpp::List screening_model_4_likes_loglin_grad(
   Rcpp::List inputs,
   double A = -0.1,
   double B = 1e-4,
   double delta = 1e-4,
   double rate = 0.1,
   double beta0 = 3.2892,
   double beta1 = -0.5533,
   double b0_psa = -1.6094,
   double b1_psa = 0.0200,
   double b2_psa = 0.1094,
   double sigma_psa = 0.2879,
   double PrFalseNegBx = 0.0,
   double tol = 1e-6,
   int n_threads = 0) {
 
 using cfaad::Number;
 
 size_t n_obs = inputs.size();
 std::vector<double> likes_val(n_obs);
 Rcpp::NumericMatrix gradients(n_obs, 11);
 
 // Extract C++ data before OpenMP to avoid R API use inside worker threads
 struct SubjectData {
   double t;
   int type;
   std::vector<double> ti;
   std::vector<double> yi;
   std::vector<int> bxi;
 };
 
 std::vector<SubjectData> data(n_obs);
 for (size_t i = 0; i < n_obs; i++) {
   Rcpp::List input = inputs(i);
   data[i].t   = Rcpp::as<double>(input("t"));
   data[i].type = Rcpp::as<int>(input("type"));
   data[i].ti  = Rcpp::as<std::vector<double>>(input("ti"));
   data[i].yi  = Rcpp::as<std::vector<double>>(input("yi"));
   data[i].bxi = Rcpp::as<std::vector<int>>(input("bxi"));
 }
 
#ifdef _OPENMP
 if (n_threads <= 0) {
   n_threads = omp_get_max_threads();
 }
#else
 n_threads = 1;
#endif
 
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
// Thread-local tape prevents data races and tape growth across subjects
cfaad::Tape local_tape;
Number::tape = &local_tape;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
for (size_t i = 0; i < n_obs; ++i) {
  
  // Reuse tape memory for each subject
  Number::tape->rewind();
  
  Number A_n(A);                         A_n.putOnTape();
  Number B_n(B);                         B_n.putOnTape();
  Number delta_n(delta);                 delta_n.putOnTape();
  Number rate_n(rate);                   rate_n.putOnTape();
  Number beta0_n(beta0);                 beta0_n.putOnTape();
  Number beta1_n(beta1);                 beta1_n.putOnTape();
  Number b0_psa_n(b0_psa);               b0_psa_n.putOnTape();
  Number b1_psa_n(b1_psa);               b1_psa_n.putOnTape();
  Number b2_psa_n(b2_psa);               b2_psa_n.putOnTape();
  Number sigma_psa_n(sigma_psa);         sigma_psa_n.putOnTape();
  Number PrFalseNegBx_n(PrFalseNegBx);   PrFalseNegBx_n.putOnTape();
  
  screening::ScreeningModel4<
    std::function<Number(double)>,
    std::function<Number(double)>,
    std::function<Number(double)>,
    std::function<Number(double)>,
    std::function<Number(double)>,
    std::function<Number(double, double, Number)>,
    Number
  > m(
      [&](double u) -> Number {
        return dMVK_t<Number>(u, A_n, B_n, delta_n);
      },
      [&](double u) -> Number {
        return pMVK_t<Number>(u, A_n, B_n, delta_n, false);
      },
      [&](double u) -> Number {
        return dexp_t<Number>(u, rate_n);
      },
      [&](double u) -> Number {
        return pexp_t<Number>(u, rate_n, false);
      },
      [&](double y) -> Number {
        return Number(1.0) /
          (Number(1.0) + cfaad::exp(-(beta0_n + beta1_n * std::log(y))));
      },
      [&](double y, double age, Number x) -> Number {
        if (y <= 0.0) return Number(0.0);
        
        double years_after_onset = std::max(0.0, age - screening::as_double(x));
        Number mu =
          b0_psa_n +
          b1_psa_n * (age - 35.0) +
          b2_psa_n * years_after_onset;
        
        Number z = (Number(std::log(y)) - mu) / sigma_psa_n;
        
        return (Number(1.0) /
                (Number(y) * sigma_psa_n * std::sqrt(2.0 * M_PI))) *
                  cfaad::exp(Number(-0.5) * z * z);
      },
      PrFalseNegBx_n,
      tol
  );
  
  m.update(data[i].ti.data(), data[i].ti.size(),
           data[i].yi.data(), data[i].yi.size(),
           data[i].bxi.data(), data[i].bxi.size());
  
  Number res;
  if (data[i].type == 1) {
    res = m.like_neg_screening(data[i].t);
  } else if (data[i].type == 2) {
    res = m.like_screen_detected_cancer(data[i].t);
  } else if (data[i].type == 3) {
    res = m.like_interval_cancer(data[i].t);
  } else {
    res = Number(-1.0);
  }
  
  // Exact reverse pass for this subject only
  res.propagateToStart();
  
  likes_val[i]    = res.value();
  gradients(i, 0) = A_n.adjoint();
  gradients(i, 1) = B_n.adjoint();
  gradients(i, 2) = delta_n.adjoint();
  gradients(i, 3) = rate_n.adjoint();
  gradients(i, 4) = beta0_n.adjoint();
  gradients(i, 5) = beta1_n.adjoint();
  gradients(i, 6) = b0_psa_n.adjoint();
  gradients(i, 7) = b1_psa_n.adjoint();
  gradients(i, 8) = b2_psa_n.adjoint();
  gradients(i, 9) = sigma_psa_n.adjoint();
  gradients(i,10) = PrFalseNegBx_n.adjoint();
}

Number::tape->clear();
}

Rcpp::colnames(gradients) = Rcpp::CharacterVector::create(
"A",
"B",
"delta",
"rate",
"beta0",
"beta1",
"b0_psa",
"b1_psa",
"b2_psa",
"sigma_psa",
"PrFalseNegBx"
);

return Rcpp::List::create(
Rcpp::Named("likelihoods") = likes_val,
Rcpp::Named("gradients")   = gradients
);
}

//' Do AD likelihood and gradient calculations for ScreeningModel5
//' (MVK onset + Exp sojourn + log-linear PSA + random effects for intercept)
//' @name screening_model_5_likes_loglin_grad
//' @param inputs list of list with elements of t, type, ti, yi, and bxi
//' @param A MVK parameter A (active for AD)
//' @param B MVK parameter B (active for AD)
//' @param delta MVK parameter delta (active for AD)
//' @param rate Exponential rate for clinical diagnosis (active for AD)
//' @param beta0 intercept for logistic no-biopsy model (active for AD)
//' @param beta1 slope of log(yi) for logistic no-biopsy model (active for AD)
//' @param mu_b0 mean intercept for log-linear PSA model (active for AD)
//' @param sigma_b0 standard deviation of intercept for log-linear PSA model (active for AD)
//' @param b1_psa age slope for log-linear PSA model (active for AD)
//' @param b2_psa slope increment after onset for log-linear PSA model (active for AD)
//' @param sigma_psa standard deviation of log(PSA) (active for AD)
//' @param PrFalseNegBx probability of a false negative biopsy (active for AD)
//' @param tol double for numeric integration tolerance
//' @param gh_nodes vector of Gauss-Hermite nodes
//' @param gh_weights vector of Gauss-Hermite weights
//' @param n_threads number of threads to use (default = 0, auto-detects)
//' @return list containing likelihoods and gradients
//' @export
// [[Rcpp::export]]
Rcpp::List screening_model_5_likes_loglin_grad(
   Rcpp::List inputs,
   double A = -0.1, double B = 1e-4, double delta = 1e-4, double rate = 0.1,
   double beta0 = 3.2892, double beta1 = -0.5533,
   double mu_b0 = -1.6094, double sigma_b0 = 0.2383,
   double b1_psa = 0.0200, double b2_psa = 0.1094, double sigma_psa = 0.2879,
   double PrFalseNegBx = 0.0, double tol = 1e-6,
   Rcpp::Nullable<Rcpp::NumericVector> gh_nodes = R_NilValue,
   Rcpp::Nullable<Rcpp::NumericVector> gh_weights = R_NilValue,
   int n_threads = 0) {
 
 using cfaad::Number;
 
 std::vector<double> nodes_vec;
 std::vector<double> weights_vec;
 
 if (gh_nodes.isNotNull() && gh_weights.isNotNull()) {
   nodes_vec = Rcpp::as<std::vector<double>>(gh_nodes);
   weights_vec = Rcpp::as<std::vector<double>>(gh_weights);
 } else {
   Rcpp::stop("gh_nodes and gh_weights must be provided for Model 5.");
 }
 
 size_t n_obs = inputs.size();
 std::vector<double> likes_val(n_obs);
 Rcpp::NumericMatrix gradients(n_obs, 12); // Now 12 parameters
 
 struct SubjectData {
   double t; int type; std::vector<double> ti; std::vector<double> yi; std::vector<int> bxi;
 };
 
 std::vector<SubjectData> data(n_obs);
 for (size_t i = 0; i < n_obs; i++) {
   Rcpp::List input = inputs(i);
   data[i].t   = Rcpp::as<double>(input("t"));
   data[i].type = Rcpp::as<int>(input("type"));
   data[i].ti  = Rcpp::as<std::vector<double>>(input("ti"));
   data[i].yi  = Rcpp::as<std::vector<double>>(input("yi"));
   data[i].bxi = Rcpp::as<std::vector<int>>(input("bxi"));
 }
 
#ifdef _OPENMP
 if (n_threads <= 0) n_threads = omp_get_max_threads();
#else
 n_threads = 1;
#endif
 
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
cfaad::Tape local_tape;
Number::tape = &local_tape;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
for (size_t i = 0; i < n_obs; ++i) {
  Number::tape->rewind();
  
  Number A_n(A);                         A_n.putOnTape();
  Number B_n(B);                         B_n.putOnTape();
  Number delta_n(delta);                 delta_n.putOnTape();
  Number rate_n(rate);                   rate_n.putOnTape();
  Number beta0_n(beta0);                 beta0_n.putOnTape();
  Number beta1_n(beta1);                 beta1_n.putOnTape();
  Number mu_b0_n(mu_b0);                 mu_b0_n.putOnTape();
  Number sigma_b0_n(sigma_b0);           sigma_b0_n.putOnTape();
  Number b1_psa_n(b1_psa);               b1_psa_n.putOnTape();
  Number b2_psa_n(b2_psa);               b2_psa_n.putOnTape();
  Number sigma_psa_n(sigma_psa);         sigma_psa_n.putOnTape();
  Number PrFalseNegBx_n(PrFalseNegBx);   PrFalseNegBx_n.putOnTape();
  
  screening::ScreeningModel5<
    std::function<Number(double)>, std::function<Number(double)>,
    std::function<Number(double)>, std::function<Number(double)>,
    std::function<Number(double)>, std::function<Number(double, double, Number, Number)>,
    Number> m(
        [&](double u) -> Number { return dMVK_t<Number>(u, A_n, B_n, delta_n); },
        [&](double u) -> Number { return pMVK_t<Number>(u, A_n, B_n, delta_n, false); },
        [&](double u) -> Number { return dexp_t<Number>(u, rate_n); },
        [&](double u) -> Number { return pexp_t<Number>(u, rate_n, false); },
        [&](double y) -> Number { return Number(1.0) / (Number(1.0) + cfaad::exp(-(beta0_n + beta1_n * std::log(y)))); },
        [&](double y, double age, Number x, Number b0) -> Number {
          if (y <= 0.0) return Number(0.0);
          double years_after_onset = std::max(0.0, age - screening::as_double(x));
          Number mu = b0 + b1_psa_n * (age - 35.0) + b2_psa_n * years_after_onset;
          Number z = (Number(std::log(y)) - mu) / sigma_psa_n;
          return (Number(1.0) / (Number(y) * sigma_psa_n * std::sqrt(2.0 * M_PI))) * cfaad::exp(Number(-0.5) * z * z);
        },
        PrFalseNegBx_n, mu_b0_n, sigma_b0_n, nodes_vec, weights_vec, tol
    );
  
  m.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size(), data[i].bxi.data(), data[i].bxi.size());
  
  Number res(0.0);
  for(size_t k = 0; k < nodes_vec.size(); ++k) {
    Number b0_k = mu_b0_n + sigma_b0_n * Number(1.4142135623730951 * nodes_vec[k]);
    Number cond_L(0.0);
    
    if (data[i].type == 1) {
      cond_L = m.like_neg_screening_cond(data[i].t, b0_k);
    } else if (data[i].type == 2) {
      cond_L = m.like_screen_detected_cancer_cond(data[i].t, b0_k);
    } else if (data[i].type == 3) {
      cond_L = m.like_interval_cancer_cond(data[i].t, b0_k);
    } else {
      cond_L = Number(-1.0);
    }
    
    res += cond_L * Number(weights_vec[k] * 0.5641895835477563);
  }
  
  res.propagateToStart();
  
  likes_val[i]    = res.value();
  gradients(i, 0) = A_n.adjoint();
  gradients(i, 1) = B_n.adjoint();
  gradients(i, 2) = delta_n.adjoint();
  gradients(i, 3) = rate_n.adjoint();
  gradients(i, 4) = beta0_n.adjoint();
  gradients(i, 5) = beta1_n.adjoint();
  gradients(i, 6) = mu_b0_n.adjoint();
  gradients(i, 7) = sigma_b0_n.adjoint();
  gradients(i, 8) = b1_psa_n.adjoint();
  gradients(i, 9) = b2_psa_n.adjoint();
  gradients(i,10) = sigma_psa_n.adjoint();
  gradients(i,11) = PrFalseNegBx_n.adjoint();
}

Number::tape->clear();
}

Rcpp::colnames(gradients) = Rcpp::CharacterVector::create(
"A", "B", "delta", "rate", "beta0", "beta1", "mu_b0", "sigma_b0", "b1_psa", "b2_psa", "sigma_psa", "PrFalseNegBx"
);

return Rcpp::List::create(
Rcpp::Named("likelihoods") = likes_val,
Rcpp::Named("gradients")   = gradients
);
}