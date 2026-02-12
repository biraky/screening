#include <Rcpp.h>
#include "screening.h"

#define _USE_MATH_DEFINES // for pi
#include <cmath>


//' Do predictions for ScreeningModel1
//' @name ScreeningModel1
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
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
//' @param shape Weibull shape for onset
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
//' @param shape Weibull shape for onset
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
//' @param shape Weibull shape for onset
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
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
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
//' @param shape Weibull shape for onset
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
 
 // Note: plnorm passed with false creates the Survival function S(t)
 screening::ScreeningModel3 m([&](double u){ return dMVK(u, A, B, delta);},
                              [&](double u){ return pMVK(u, A, B, delta, 0);},
                              [&](double u){ return dlnorm(u, mulog, sdlog); },
                              [&](double u){ return plnorm(u, mulog, sdlog, false); },
                              [&](double y){ return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y))));},
                              PrFalseNegBx,
                              tol);
 
 return m.likes(inputs, 1e-12, return_type, w, left_trunc, incidence);
}