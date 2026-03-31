#ifndef SCREENING_SCREENING_H
#define SCREENING_SCREENING_H = 1

// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <array>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <omp.h>
#define _USE_MATH_DEFINES
#include <cmath>

// AD compilers regquired by Boost integration====================================================
#include "cfaad/AAD.h"

namespace cfaad {
// print a cfaad::Number (required by Boost Error Handling)
inline std::ostream& operator<<(std::ostream& os, const Number& n) {
  os << n.value();
  return os;
}

// absolute value of cfaad types (required by Boost Quadrature)
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

// Boost Quadrature needs numeric limits to compile generic AD types.
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
//================================================================================================================


namespace screening {

// as_double: depending on whether T is double or cfaad::Number
inline double as_double(double x) { return x; }
template <class T>
inline double as_double(const T& x) { return x.value(); }

// -------------------------------------------------------------
// Statistical Distributions (Standard and AD)
// -------------------------------------------------------------
namespace dist {
// --- Weibull ---
template <typename T>
inline T dweibull_t(double x, T shape, T scale) {
  if (x < 0) return T(0.0);
  T xl = T(x) / scale;
  return (shape / scale) * cfaad::pow(xl, shape - T(1.0)) * cfaad::exp(-cfaad::pow(xl, shape));
}
template <>
inline double dweibull_t<double>(double x, double shape, double scale) {
  if (x < 0) return 0.0;
  double xl = x / scale;
  return (shape / scale) * std::pow(xl, shape - 1.0) * std::exp(-std::pow(xl, shape));
}

template <typename T>
inline T pweibull_t(double x, T shape, T scale, bool lower_tail = false) {
  if (x < 0) {
    if (lower_tail) return T(0.0);
    else return T(1.0);
  }
  T val = cfaad::exp(-cfaad::pow(T(x) / scale, shape));
  if (lower_tail) {
    return T(1.0) - val;
  } else {
    return val;
  }
}
template <>
inline double pweibull_t<double>(double x, double shape, double scale, bool lower_tail) {
  if (x < 0) return lower_tail ? 0.0 : 1.0;
  return lower_tail ? 1.0 - std::exp(-std::pow(x / scale, shape)) : std::exp(-std::pow(x / scale, shape));
}

// --- MVK ---
template <typename T>
inline T dMVK_t(double t, T A, T B, T delta) {
  T P = (cfaad::exp((B - A) * t) - T(1.0)) * cfaad::exp(B * delta * t) * cfaad::pow(B - A, delta);
  T Q = cfaad::pow(B * cfaad::exp((B - A) * t) - A, T(1.0) + delta);
  T val = -delta * A * B * P / Q;
  return val;
}
template <>
inline double dMVK_t<double>(double t, double A, double B, double delta) {
  double P = std::expm1((B - A) * t) * std::exp(B * delta * t) * std::pow(B - A, delta);
  double Q = std::pow(B * std::exp((B - A) * t) - A, 1.0 + delta);
  double val = -delta * A * B * P / Q;
  if (!std::isfinite(val)) val = 0.0;
  return val;
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
template <>
inline double pMVK_t<double>(double t, double A, double B, double delta, bool lower_tail) {
  double logS = delta * (std::log(B - A) + B * t - std::log(B * std::exp((B - A) * t) - A));
  return lower_tail ? -std::expm1(logS) : std::exp(logS);
}

// --- Beard ---
template <typename T>
inline T dBeard_t(double t, T alpha, T beta, T kappa) {
  if (t < 0) return T(0.0);
  T e_bt = cfaad::exp(beta * t);
  T h_t = (alpha * e_bt) / (T(1.0) + kappa * e_bt);
  T S_t = cfaad::pow((T(1.0) + kappa * e_bt) / (T(1.0) + kappa), -alpha / (kappa * beta));
  return h_t * S_t;
}
template <>
inline double dBeard_t<double>(double t, double alpha, double beta, double kappa) {
  if (t < 0) return 0.0;
  double e_bt = std::exp(beta * t);
  double h_t = (alpha * e_bt) / (1.0 + kappa * e_bt);
  double S_t = std::pow((1.0 + kappa * e_bt) / (1.0 + kappa), -alpha / (kappa * beta));
  return h_t * S_t;
}

template <typename T>
inline T pBeard_t(double t, T alpha, T beta, T kappa, bool lower_tail = true) {
  if (t < 0) return lower_tail ? T(0.0) : T(1.0);
  T e_bt = cfaad::exp(beta * t);
  T S_t = cfaad::pow((T(1.0) + kappa * e_bt) / (T(1.0) + kappa), -alpha / (kappa * beta));
  if (lower_tail) {
    return T(1.0) - S_t;
  } else {
    return S_t;
  }
}
template <>
inline double pBeard_t<double>(double t, double alpha, double beta, double kappa, bool lower_tail) {
  if (t < 0) return lower_tail ? 0.0 : 1.0;
  double e_bt = std::exp(beta * t);
  double S_t = std::pow((1.0 + kappa * e_bt) / (1.0 + kappa), -alpha / (kappa * beta));
  return lower_tail ? 1.0 - S_t : S_t;
}

// --- Exponential ---
template <typename T>
inline T dexp_t(double t, T rate) {
  return (t < 0) ? T(0.0) : rate * cfaad::exp(-rate * t);
}
template <>
inline double dexp_t<double>(double t, double rate) {
  return (t < 0) ? 0.0 : rate * std::exp(-rate * t);
}

template <typename T>
inline T pexp_t(double t, T rate, bool lower_tail = true) {
  if (t < 0) {
    if (lower_tail) return T(0.0);
    else return T(1.0);
  }
  T val = cfaad::exp(-rate * t);
  if (lower_tail) {
    return T(1.0) - val;
  } else {
    return val;
  }
}
template <>
inline double pexp_t<double>(double t, double rate, bool lower_tail) {
  if (t < 0) return lower_tail ? 0.0 : 1.0;
  return lower_tail ? 1.0 - std::exp(-rate * t) : std::exp(-rate * t);
}

// --- Lognormal (Standard only, AD unmapped for erfc) ---
inline double dlnorm_t(double t, double mulog, double sdlog) {
  return (t <= 0) ? 0.0 : (1.0 / (t * sdlog * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * std::pow((std::log(t) - mulog) / sdlog, 2.0));
}
inline double plnorm_t(double t, double mulog, double sdlog, bool lower_tail = true) {
  return (t <= 0) ? (lower_tail ? 0.0 : 1.0) : 0.5 * std::erfc(((lower_tail ? -1.0 : 1.0) * (std::log(t) - mulog)) / (sdlog * std::sqrt(2.0)));
}
}

// -------------------------------------------------------------
// Core Screening Models
// -------------------------------------------------------------

// Abstract templated class for a base screening model
template<class T1, class T2, class T3, class T4, class T_out = double>
class AbstractScreeningModel {
public:
  T1 f1; // density for onset
  T2 S1; // survival for onset
  T3 f2; // density for clinical symptoms
  T4 S2; // survival for clinical symptoms
  std::vector<double> ti, tj; // episodes times
  double tol; // integration tolerance
  size_t fulln, n; // number of episodes
  T_out error; // used in the integrations
  int offset; 
  
  AbstractScreeningModel(T1 f1, T2 S1, T3 f2, T4 S2, double tol = 1.0e-6)
    : f1(f1), S1(S1), f2(f2), S2(S2), tol(tol), fulln(0) { }
  
  virtual ~AbstractScreeningModel() { }
  
  virtual T_out prod_beta(size_t i, size_t j, bool detected = false) = 0;
  virtual std::vector<T_out> likes(Rcpp::List inputs, double eps=1.0e-12) = 0;
  
  void setup(double t) {
    size_t i;
    tj.resize(0);
    tj.push_back(0.0);
    for (i=0; i<fulln && ti[i]<t; i++) {
      tj.push_back(ti[i]);
    }
    offset = (i+1==fulln && ti[i]==t) ? 1 : 0;
    tj.push_back(t);
    n = tj.size()-2; 
  }
  
  void update(std::vector<double> ti) {
    this->ti=ti;
    fulln=ti.size();
  }
  
  void update(const double* ti_ptr, size_t n_ti) {
    this->ti.assign(ti_ptr, ti_ptr + n_ti);
    fulln = n_ti;
  }
  
  T_out X(double t, bool reset = true) {
    if (reset) setup(t);
    return S1(t);
  }
  
  T_out Y(double t, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) setup(t);
    T_out value(0.0);
    for (size_t i=0; i<=n; i++) {
      T_out K = prod_beta(i, n+offset);
      auto fn = [&](T_out x_ad) -> T_out { 
        double x = as_double(x_ad);
        return f1(x)*S2(t-x)*K; 
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(tj[i]), T_out(tj[i+1]), 5, T_out(tol), &error);
    }
    return value;
  }
  
  T_out I(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) setup(t);
    T_out value(0.0);
    for (size_t i=0; i<=n; i++) {
      T_out K = prod_beta(i, n+offset);
      auto fn = [&](T_out x_ad) -> T_out { 
        double x = as_double(x_ad);
        return f1(x)*f2(t-x)*K; 
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(tj[i]), T_out(tj[i+1]), 5, T_out(tol), &error);
    }
    return value;
  }
  
  T_out Z(double t, bool reset=true, bool simple=true) {
    using namespace boost::math::quadrature;
    if (simple) return (T_out(1.0) - (X(t)+Y(t)));
    if (reset) setup(t);
    T_out value(0.0), error2;
    for (size_t i=0; i<=n; i++) { 
      for (size_t j=i; j<=n; j++) { 
        T_out K = prod_beta(i,j);
        auto fn = [&](T_out s_ad) -> T_out {
          double s = as_double(s_ad);
          auto inner = [&](T_out u_ad) -> T_out { 
            double u = as_double(u_ad);
            return f1(s)*K*f2(u-s); 
          };
          return gauss_kronrod<T_out, 15>::integrate(inner, T_out(std::max(s,tj[j])), T_out(tj[j+1]), 5, T_out(tol), &error2);
        };
        value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(tj[i]), T_out(tj[i+1]), 5, T_out(tol), &error);
      }
    }
    for (size_t i=0; i<=n; i++) { 
      for (size_t j=i+1; j<=n+offset; j++) { 
        T_out K = prod_beta(i,j,true);
        auto fn = [&](T_out u_ad) -> T_out { 
          double u = as_double(u_ad);
          return f1(u)*S2(tj[j]-u)*K; 
        };
        value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(tj[i]), T_out(tj[i+1]), 5, T_out(tol), &error);
      }
    }
    return value;
  }
  
  Rcpp::DataFrame predictions(std::vector<double> t, bool simple = true) {
    using namespace Rcpp;
    NumericMatrix out(t.size(), 4);
    for (size_t i=0; i<t.size(); i++) {
      out(i,0) = static_cast<double>(X(t[i]));  
      out(i,1) = static_cast<double>(Y(t[i]));  
      out(i,2) = static_cast<double>(Z(t[i], true, simple));  
      out(i,3) = static_cast<double>(I(t[i]));  
    }
    return DataFrame::create(_("t")=t,
                             _("X")=out.column(0),
                             _("Y")=out.column(1),
                             _("Z")=out.column(2),
                             _("I")=out.column(3));
  }
};


// ScreeningModel1
template<class T1, class T2, class T3, class T4, class T_out = double>
class ScreeningModel1 : public AbstractScreeningModel<T1,T2,T3,T4,T_out> {
public:
  T_out beta;
  ScreeningModel1(T1 f1, T2 S1, T3 f2, T4 S2, T_out beta, double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4,T_out>(f1,S1,f2,S2, tol),
    beta(beta) { }
  
  T_out prod_beta(size_t i, size_t j, bool detected = false) {
    T_out value(1.0);
    for (size_t k=i; k<j; k++) value *= (detected && k+1==j ? T_out(1.0)-beta : beta);
    return value;
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps=1.0e-12) {
    using Rcpp::as;
    size_t n_obs = inputs.size();
    std::vector<T_out> out(n_obs);
    
    struct SubjectData {
      double t;
      int type;
      std::vector<double> ti;
    };
    
    std::vector<SubjectData> data(n_obs);
    for (size_t i = 0; i < n_obs; i++) {
      Rcpp::List input = inputs(i);
      data[i].t = as<double>(input("t"));
      data[i].type = as<int>(input("type"));
      data[i].ti = as<std::vector<double>>(input("ti"));
    }
    
#pragma omp parallel if(std::is_same<T_out, double>::value)
{
  ScreeningModel1 local_model = *this;
  
#pragma omp for schedule(static)
  for (int i = 0; i < (int)n_obs; i++) {
    local_model.update(data[i].ti.data(), data[i].ti.size());
    
    if (data[i].type == 1) {  
      out[i] = local_model.X(data[i].t) + local_model.Y(data[i].t);
    } 
    else if (data[i].type == 2) {  
      out[i] = local_model.Y(data[i].t-eps) * (T_out(1.0) - local_model.beta);
    } 
    else if (data[i].type == 3) {  
      out[i] = local_model.I(data[i].t);
    } 
    else {
      out[i] = T_out(-1.0);  
    }
  }
}
return out;
  }
};


// ScreeningModel2
template<class T1, class T2, class T3, class T4, class T5, class T_out = double>
class ScreeningModel2 : public AbstractScreeningModel<T1,T2,T3,T4, T_out> {
public:
  std::vector<double> yi;
  T5 PrFalseNeg;
  ScreeningModel2(T1 f1, T2 S1, T3 f2, T4 S2, T5 PrFalseNeg,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4, T_out>(f1,S1,f2,S2, tol),     
    PrFalseNeg(PrFalseNeg) {}
  void update(std::vector<double> ti,
              std::vector<double> yi) {
    AbstractScreeningModel<T1,T2,T3,T4, T_out>::update(ti);
    this->yi=yi;
  }
  
  void update(const double* ti_ptr, size_t ti_n, 
              const double* yi_ptr, size_t yi_n) {
    AbstractScreeningModel<T1,T2,T3,T4, T_out>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
  }
  
  T_out prod_beta(size_t i, size_t j, bool detected = false) {
    T_out value(1.0);
    for (size_t k=i; k<j; k++) {
      value *= (detected && k+1==j ? T_out(1.0)-PrFalseNeg(yi[k]) : PrFalseNeg(yi[k]));
    }
    return value;
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps = 1.0e-12) {
    using Rcpp::as;
    size_t n_obs = inputs.size();
    std::vector<T_out> out(n_obs);
    
    struct SubjectData {
      double t;
      int type;
      std::vector<double> ti;
      std::vector<double> yi;
    };
    
    std::vector<SubjectData> data(n_obs);
    for (size_t i = 0; i < n_obs; i++) {
      Rcpp::List input = inputs(i);
      data[i].t = as<double>(input("t"));
      data[i].type = as<int>(input("type"));
      data[i].ti = as<std::vector<double>>(input("ti"));
      data[i].yi = as<std::vector<double>>(input("yi"));
    }
    
#pragma omp parallel if(std::is_same<T_out, double>::value)
{
  ScreeningModel2 local_model = *this;
  
#pragma omp for schedule(static)
  for (int i = 0; i < (int)n_obs; i++) {
    local_model.update(data[i].ti.data(), data[i].ti.size(),
                       data[i].yi.data(), data[i].yi.size());
    
    if (data[i].type == 1) { 
      out[i] = local_model.X(data[i].t) + local_model.Y(data[i].t);
    } 
    else if (data[i].type == 2) { 
      if (local_model.fulln > 0) {
        out[i] = local_model.Y(data[i].t-eps) * (T_out(1.0) - local_model.PrFalseNeg(local_model.yi[local_model.n-1]));
      } else {
        out[i] = T_out(0.0); 
      }
    }
    else if (data[i].type == 3) { 
      out[i] = local_model.I(data[i].t);
    } 
    else {
      out[i] = T_out(-1.0);  
    }
  }
}
return out;
  }
};


// ScreeningModel3
template<class T1, class T2, class T3, class T4, class T5, class T_out = double>
class ScreeningModel3 : public AbstractScreeningModel<T1,T2,T3,T4, T_out> {
public:
  std::vector<double> yi; 
  std::vector<int> bxi; 
  T5 PrNoBx;
  T_out PrFalseNegBx;
  
  ScreeningModel3(T1 f1, T2 S1, T3 f2, T4 S2,
                  T5 PrNoBx,
                  T_out PrFalseNegBx,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4, T_out>(f1,S1,f2,S2,tol),
    PrNoBx(PrNoBx), PrFalseNegBx(PrFalseNegBx) { }
  
  void update(std::vector<double> ti, std::vector<double> yi, std::vector<int> bxi) {
    AbstractScreeningModel<T1,T2,T3,T4, T_out>::update(ti);
    this->yi=yi;
    this->bxi=bxi;
  }
  
  void update(const double* ti_ptr, size_t ti_n, 
              const double* yi_ptr, size_t yi_n, 
              const int* bxi_ptr, size_t bxi_n) {
    AbstractScreeningModel<T1,T2,T3,T4, T_out>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
    this->bxi.assign(bxi_ptr, bxi_ptr + bxi_n);
  }
  
  T_out prod_bx(size_t i, size_t j) {
    T_out value(1.0);
    for (size_t k=i; k<j; k++)
      value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (T_out(1.0)-PrNoBx(yi[k])));
    return value;
  }
  
  T_out prod_beta(size_t i, size_t j, bool detected = false) {
    T_out value(1.0);
    for (size_t k=i; k<j; k++)
      value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (T_out(1.0)-PrNoBx(yi[k]))*(detected && k+1==j ? (T_out(1.0) - PrFalseNegBx) : PrFalseNegBx));
    return value;
  }
  
  T_out like_neg_screening(double s, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(s);
    T_out value = this->S1(s)*prod_bx(0,this->n);
    for (size_t i=0; i<=this->n; i++) {
      T_out K = prod_bx(0,i) * prod_beta(i,this->n);
      auto fn = [&](T_out x_ad) -> T_out {
        double x = as_double(x_ad);
        return this->f1(x)*this->S2(s-x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i+1]),
                                                   5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_screen_detected_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i=0; i<=this->n; i++) {
      T_out K = prod_bx(0,i) * prod_beta(i,this->n+this->offset,true);
      auto fn = [&](T_out x_ad) -> T_out {
        double x = as_double(x_ad);
        return this->f1(x)*this->S2(t-x)*K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i+1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_interval_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i=0; i<=this->n; i++) {
      T_out K = prod_bx(0,i)*prod_beta(i,this->n+this->offset);
      auto fn = [&](T_out x_ad) -> T_out {
        double x = as_double(x_ad);
        return this->f1(x)*this->f2(t-x)*K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i+1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps=1.0e-12) override {
    return likes(inputs, eps, "", {}); 
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps=1.0e-12,
                           std::string return_type = "",
                           std::vector<double> weights = {},
                           bool left_trunc = false,
                           Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
    
    bool weighted_ll = return_type == "weighted_ll";
    
    std::vector<double> inc_years;
    std::vector<std::vector<double>> inc_rates;
    int n_inc_years = 0;
    
    if (left_trunc && incidence.isNotNull()){
      Rcpp::DataFrame df(incidence);
      inc_years = Rcpp::as<std::vector<double>>(df["year"]);
      n_inc_years = inc_years.size();
      
      std::vector<std::string> age_cols = {
        "<40", "40-44", "45-49", "50-54", "55-59", 
        "60-64", "65-69", "70-74", "75-79", "80-84", "85+"
      };
      
      int n_age_grps = age_cols.size();
      inc_rates.resize(n_inc_years, std::vector<double>(n_age_grps)); 
      
      for(int j = 0; j < n_age_grps; ++j) {
        std::vector<double> current_col = Rcpp::as<std::vector<double>>(df[age_cols[j]]);
        for(int i = 0; i < n_inc_years; ++i) {
          inc_rates[i][j] = current_col[i] / 100000.0; 
        }
      }
    }
    
    struct SubjectData {
      double t;
      int type;
      std::vector<double> ti; 
      std::vector<double> yi; 
      std::vector<int> bxi;
      double dob;
    };
    
    size_t n_obs = inputs.size();
    
    if(!weights.empty() && weights.size() != n_obs) {
      Rcpp::stop("The size of weights must be equal to the size of inputs.");
    }
    
    std::vector<SubjectData> data(n_obs);
    std::vector<T_out> out(n_obs);
    
    for(size_t i = 0; i < n_obs; ++i) {
      Rcpp::List subject = inputs[i];
      data[i].t = Rcpp::as<double>(subject["t"]);
      data[i].type = Rcpp::as<int>(subject["type"]);
      data[i].dob = Rcpp::as<double>(subject["dob"]);
      
      if(data[i].type >3) {
        Rcpp::stop("Only 3 screen types are supported");
      }
      
      data[i].ti = Rcpp::as<std::vector<double>>(subject["ti"]);
      data[i].yi = Rcpp::as<std::vector<double>>(subject["yi"]);
      data[i].bxi = Rcpp::as<std::vector<int>>(subject["bxi"]);
    }
    
#pragma omp parallel if(std::is_same<T_out, double>::value)
{
  using namespace boost::math::quadrature;
  ScreeningModel3 local_model = *this;
  
#pragma omp for schedule(static)
  for (int i = 0; i < (int)n_obs; i++) {
    local_model.update(data[i].ti.data(), data[i].ti.size(), 
                       data[i].yi.data(), data[i].yi.size(),
                       data[i].bxi.data(), data[i].bxi.size());
    
    if (data[i].type == 1) {
      out[i] = local_model.like_neg_screening(data[i].t);
    } else if (data[i].type == 2) { 
      out[i] = local_model.like_screen_detected_cancer(data[i].t);
    } else if (data[i].type == 3) { 
      out[i] = local_model.like_interval_cancer(data[i].t);
    } else {
      out[i] = T_out(-1.0);
    }
    
    if(left_trunc) {
      double cum_haz = 0.0;
      for(int k = 0; k < n_inc_years; k++) {
        double age_at_year = inc_years[k] - data[i].dob;
        if (age_at_year < 0) continue;
        int age_idx = 0;
        
        if (age_at_year < 40.0) {
          age_idx = 0;
        } else if (age_at_year >= 85.0) {
          age_idx = 10;
        } else {
          age_idx = (int)((age_at_year - 40.0) / 5.0) + 1;
        }
        cum_haz += inc_rates[k][age_idx];
      }
      double X_Y_1997_2006 = std::exp(-cum_haz);
      
      double date_1997_days = 9862.0;
      double age_1997 = (date_1997_days - data[i].dob) / 365.25;
      
      T_out X_Y_0_1997(1.0);
      if (age_1997 > 0) {
        auto fn = [&](T_out x_ad) -> T_out {
          double x = as_double(x_ad);
          return local_model.f1(x) * local_model.S2(age_1997 - x);
        };
        X_Y_0_1997 = local_model.S1(age_1997) + 
          boost::math::quadrature::gauss_kronrod<T_out, 15>::integrate(fn, T_out(0.0), T_out(age_1997), 5,
                                                                       T_out(local_model.tol),
                                                                       &local_model.error);
      }
      out[i] = out[i] / X_Y_0_1997 / T_out(X_Y_1997_2006);
    }
    if(weighted_ll) {
      using std::log;
      out[i] = T_out(weights[i]) * log(out[i]);
    }
  }
}

return out;
  }
};

// ScreeningModel4
template<class T1, class T2, class T3, class T4, class T5, class T6, class T_out = double>
class ScreeningModel4 : public AbstractScreeningModel<T1,T2,T3,T4,T_out> {
public:
  std::vector<double> yi;
  std::vector<int> bxi;
  T5 PrNoBx;
  T6 biomarker_den;
  T_out PrFalseNegBx;
  
  ScreeningModel4(T1 f1, T2 S1, T3 f2, T4 S2,
                  T5 PrNoBx,
                  T6 biomarker_den,
                  T_out PrFalseNegBx,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4,T_out>(f1, S1, f2, S2, tol),
    PrNoBx(PrNoBx),
    biomarker_den(biomarker_den),
    PrFalseNegBx(PrFalseNegBx) { }
  
  void update(std::vector<double> ti,
              std::vector<double> yi,
              std::vector<int> bxi) {
    AbstractScreeningModel<T1,T2,T3,T4,T_out>::update(ti);
    this->yi = yi;
    this->bxi = bxi;
  }
  
  void update(const double* ti_ptr, size_t ti_n,
              const double* yi_ptr, size_t yi_n,
              const int* bxi_ptr, size_t bxi_n) {
    AbstractScreeningModel<T1,T2,T3,T4,T_out>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
    this->bxi.assign(bxi_ptr, bxi_ptr + bxi_n);
  }
  
  T_out prod_history(size_t i, size_t j, T_out x, bool detected = false) {
    T_out value(1.0);
    for (size_t k = 0; k < j; k++) {
      value *= biomarker_den(yi[k], this->tj[k + 1], x);
      
      T_out p_no_bx = PrNoBx(yi[k]);
      T_out p_bx    = T_out(1.0) - p_no_bx;
      
      if (k < i) {
        value *= (bxi[k] == 0 ? p_no_bx : p_bx);
      } else {
        value *= (bxi[k] == 0 ?
                    p_no_bx :
                    p_bx * (detected && k + 1 == j ? (T_out(1.0) - PrFalseNegBx) : PrFalseNegBx));
      }
    }
    return value;
  }
  
  T_out prod_beta(size_t /*i*/, size_t /*j*/, bool /*detected*/ = false) override {
    return T_out(1.0);
  }
  
  T_out like_neg_screening(double s, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(s);
    
    T_out no_onset_history(1.0);
    for (size_t k = 0; k < this->n; k++) {
      no_onset_history *= biomarker_den(yi[k], this->tj[k + 1], T_out(1.0e9));
      T_out p_no_bx = PrNoBx(yi[k]);
      no_onset_history *= (bxi[k] == 0 ? p_no_bx : (T_out(1.0) - p_no_bx));
    }
    
    T_out value = this->S1(s) * no_onset_history;
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n, x_ad, false);
        double x = as_double(x_ad);
        return this->f1(x) * this->S2(s - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_screen_detected_cancer(double t, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n + this->offset, x_ad, true);
        double x = as_double(x_ad);
        return this->f1(x) * this->S2(t - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_interval_cancer(double t, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n + this->offset, x_ad, false);
        double x = as_double(x_ad);
        return this->f1(x) * this->f2(t - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps = 1.0e-12) override {
    return likes(inputs, eps, "", {});
  }
  
  std::vector<T_out> likes(Rcpp::List inputs,
                           double eps = 1.0e-12,
                           std::string return_type = "",
                           std::vector<double> weights = {},
                           bool left_trunc = false,
                           Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
    
    bool weighted_ll = return_type == "weighted_ll";
    
    std::vector<double> inc_years;
    std::vector<std::vector<double>> inc_rates;
    int n_inc_years = 0;
    
    if (left_trunc && incidence.isNotNull()) {
      Rcpp::DataFrame df(incidence);
      inc_years = Rcpp::as<std::vector<double>>(df["year"]);
      n_inc_years = inc_years.size();
      std::vector<std::string> age_cols = {
        "<40", "40-44", "45-49", "50-54", "55-59",
        "60-64", "65-69", "70-74", "75-79", "80-84", "85+"
      };
      int n_age_grps = age_cols.size();
      inc_rates.resize(n_inc_years, std::vector<double>(n_age_grps));
      for (int j = 0; j < n_age_grps; ++j) {
        std::vector<double> current_col = Rcpp::as<std::vector<double>>(df[age_cols[j]]);
        for (int i = 0; i < n_inc_years; ++i) {
          inc_rates[i][j] = current_col[i] / 100000.0;
        }
      }
    }
    
    struct SubjectData {
      double t; int type; std::vector<double> ti; std::vector<double> yi; std::vector<int> bxi; double dob;
    };
    
    size_t n_obs = inputs.size();
    if (!weights.empty() && weights.size() != n_obs) {
      Rcpp::stop("The size of weights must be equal to the size of inputs.");
    }
    
    std::vector<SubjectData> data(n_obs);
    std::vector<T_out> out(n_obs);
    
    for (size_t i = 0; i < n_obs; ++i) {
      Rcpp::List subject = inputs[i];
      data[i].t = Rcpp::as<double>(subject["t"]);
      data[i].type = Rcpp::as<int>(subject["type"]);
      data[i].dob = Rcpp::as<double>(subject["dob"]);
      data[i].ti  = Rcpp::as<std::vector<double>>(subject["ti"]);
      data[i].yi  = Rcpp::as<std::vector<double>>(subject["yi"]);
      data[i].bxi = Rcpp::as<std::vector<int>>(subject["bxi"]);
    }
    
#pragma omp parallel if(std::is_same<T_out, double>::value)
{
  ScreeningModel4 local_model = *this;
  
#pragma omp for schedule(static)
  for (int i = 0; i < (int)n_obs; i++) {
    local_model.update(data[i].ti.data(), data[i].ti.size(),
                       data[i].yi.data(), data[i].yi.size(),
                       data[i].bxi.data(), data[i].bxi.size());
    
    if (data[i].type == 1) { out[i] = local_model.like_neg_screening(data[i].t); } 
    else if (data[i].type == 2) { out[i] = local_model.like_screen_detected_cancer(data[i].t); } 
    else if (data[i].type == 3) { out[i] = local_model.like_interval_cancer(data[i].t); } 
    else { out[i] = T_out(-1.0); }
    
    if (left_trunc) {
      double cum_haz = 0.0;
      for (int k = 0; k < n_inc_years; k++) {
        double age_at_year = inc_years[k] - data[i].dob;
        if (age_at_year < 0) continue;
        int age_idx = 0;
        if (age_at_year < 40.0) age_idx = 0;
        else if (age_at_year >= 85.0) age_idx = 10;
        else age_idx = (int)((age_at_year - 40.0) / 5.0) + 1;
        cum_haz += inc_rates[k][age_idx];
      }
      
      double X_Y_1997_2006 = std::exp(-cum_haz);
      double date_1997_days = 9862.0;
      double age_1997 = (date_1997_days - data[i].dob) / 365.25;
      T_out X_Y_0_1997(1.0);
      
      if (age_1997 > 0) {
        auto fn = [&](T_out x_ad) -> T_out {
          double x = as_double(x_ad);
          return local_model.f1(x) * local_model.S2(age_1997 - x);
        };
        X_Y_0_1997 = local_model.S1(age_1997) + boost::math::quadrature::gauss_kronrod<T_out, 15>::integrate(
          fn, T_out(0.0), T_out(age_1997), 5, T_out(local_model.tol), &local_model.error);
      }
      out[i] = out[i] / X_Y_0_1997 / T_out(X_Y_1997_2006);
    }
    if (weighted_ll) {
      using std::log;
      out[i] = T_out(weights[i]) * log(out[i]);
    }
  }
}
return out;
  }
};

// ScreeningModel5
template<class T1, class T2, class T3, class T4, class T5, class T6, class T_out = double>
class ScreeningModel5 : public AbstractScreeningModel<T1,T2,T3,T4,T_out> {
public:
  std::vector<double> yi;
  std::vector<int> bxi;
  T5 PrNoBx;
  T6 biomarker_den;
  T_out PrFalseNegBx;
  T_out mu_b0;
  T_out sigma_b0;
  std::vector<double> gh_nodes;
  std::vector<double> gh_weights;
  
  ScreeningModel5(T1 f1, T2 S1, T3 f2, T4 S2,
                  T5 PrNoBx, T6 biomarker_den, T_out PrFalseNegBx,
                  T_out mu_b0, T_out sigma_b0,
                  std::vector<double> gh_nodes, std::vector<double> gh_weights,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4,T_out>(f1, S1, f2, S2, tol),
    PrNoBx(PrNoBx), biomarker_den(biomarker_den), PrFalseNegBx(PrFalseNegBx),
    mu_b0(mu_b0), sigma_b0(sigma_b0), gh_nodes(gh_nodes), gh_weights(gh_weights) { }
  
  void update(const double* ti_ptr, size_t ti_n,
              const double* yi_ptr, size_t yi_n,
              const int* bxi_ptr, size_t bxi_n) {
    AbstractScreeningModel<T1,T2,T3,T4,T_out>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
    this->bxi.assign(bxi_ptr, bxi_ptr + bxi_n);
  }
  
  T_out prod_history(size_t i, size_t j, T_out x, T_out b0, bool detected = false) {
    T_out value(1.0);
    for (size_t k = 0; k < j; k++) {
      value *= biomarker_den(yi[k], this->tj[k + 1], x, b0);
      T_out p_no_bx = PrNoBx(yi[k]);
      T_out p_bx    = T_out(1.0) - p_no_bx;
      
      if (k < i) {
        value *= (bxi[k] == 0 ? p_no_bx : p_bx);
      } else {
        value *= (bxi[k] == 0 ? p_no_bx : p_bx * (detected && k + 1 == j ? (T_out(1.0) - PrFalseNegBx) : PrFalseNegBx));
      }
    }
    return value;
  }
  
  T_out prod_beta(size_t /*i*/, size_t /*j*/, bool /*detected*/ = false) override {
    return T_out(1.0);
  }
  
  T_out like_neg_screening_cond(double s, T_out b0, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(s);
    
    T_out no_onset_history(1.0);
    for (size_t k = 0; k < this->n; k++) {
      no_onset_history *= biomarker_den(yi[k], this->tj[k + 1], T_out(1.0e9), b0);
      T_out p_no_bx = PrNoBx(yi[k]);
      no_onset_history *= (bxi[k] == 0 ? p_no_bx : (T_out(1.0) - p_no_bx));
    }
    
    T_out value = this->S1(s) * no_onset_history;
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n, x_ad, b0, false);
        double x = as_double(x_ad);
        return this->f1(x) * this->S2(s - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_screen_detected_cancer_cond(double t, T_out b0, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n + this->offset, x_ad, b0, true);
        double x = as_double(x_ad);
        return this->f1(x) * this->S2(t - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  T_out like_interval_cancer_cond(double t, T_out b0, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    T_out value(0.0);
    for (size_t i = 0; i <= this->n; i++) {
      auto fn = [&](T_out x_ad) -> T_out {
        T_out K = prod_history(i, this->n + this->offset, x_ad, b0, false);
        double x = as_double(x_ad);
        return this->f1(x) * this->f2(t - x) * K;
      };
      value += gauss_kronrod<T_out, 15>::integrate(fn, T_out(this->tj[i]), T_out(this->tj[i + 1]), 5, T_out(this->tol), &this->error);
    }
    return value;
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps = 1.0e-12) override {
    return likes(inputs, eps, "", {});
  }
  
  std::vector<T_out> likes(Rcpp::List inputs, double eps = 1.0e-12, std::string return_type = "", std::vector<double> weights = {}, bool left_trunc = false, Rcpp::Nullable<Rcpp::DataFrame> incidence = R_NilValue) {
    bool weighted_ll = return_type == "weighted_ll";
    std::vector<double> inc_years;
    std::vector<std::vector<double>> inc_rates;
    int n_inc_years = 0;
    
    if (left_trunc && incidence.isNotNull()) {
      Rcpp::DataFrame df(incidence);
      inc_years = Rcpp::as<std::vector<double>>(df["year"]);
      n_inc_years = inc_years.size();
      std::vector<std::string> age_cols = {"<40", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"};
      int n_age_grps = age_cols.size();
      inc_rates.resize(n_inc_years, std::vector<double>(n_age_grps));
      for (int j = 0; j < n_age_grps; ++j) {
        std::vector<double> current_col = Rcpp::as<std::vector<double>>(df[age_cols[j]]);
        for (int i = 0; i < n_inc_years; ++i) { inc_rates[i][j] = current_col[i] / 100000.0; }
      }
    }
    
    struct SubjectData {
      double t; int type; double dob;
      std::vector<double> ti; std::vector<double> yi; std::vector<int> bxi;
    };
    
    size_t n_obs = inputs.size();
    if (!weights.empty() && weights.size() != n_obs) Rcpp::stop("The size of weights must be equal to the size of inputs.");
    
    std::vector<SubjectData> data(n_obs);
    std::vector<T_out> out(n_obs);
    
    for (size_t i = 0; i < n_obs; ++i) {
      Rcpp::List subject = inputs[i];
      data[i].t = Rcpp::as<double>(subject["t"]);
      data[i].type = Rcpp::as<int>(subject["type"]);
      data[i].dob = Rcpp::as<double>(subject["dob"]);
      data[i].ti  = Rcpp::as<std::vector<double>>(subject["ti"]);
      data[i].yi  = Rcpp::as<std::vector<double>>(subject["yi"]);
      data[i].bxi = Rcpp::as<std::vector<int>>(subject["bxi"]);
    }
    
#pragma omp parallel if(std::is_same<T_out, double>::value)
{
  ScreeningModel5 local_model = *this;
  
#pragma omp for schedule(static)
  for (int i = 0; i < (int)n_obs; i++) {
    local_model.update(data[i].ti.data(), data[i].ti.size(), data[i].yi.data(), data[i].yi.size(), data[i].bxi.data(), data[i].bxi.size());
    
    T_out L_i(0.0);
    for(size_t k = 0; k < local_model.gh_nodes.size(); ++k) {
      T_out b0_k = local_model.mu_b0 + local_model.sigma_b0 * T_out(1.4142135623730951 * local_model.gh_nodes[k]);
      T_out cond_L(0.0);
      
      if (data[i].type == 1) { cond_L = local_model.like_neg_screening_cond(data[i].t, b0_k); } 
      else if (data[i].type == 2) { cond_L = local_model.like_screen_detected_cancer_cond(data[i].t, b0_k); } 
      else if (data[i].type == 3) { cond_L = local_model.like_interval_cancer_cond(data[i].t, b0_k); } 
      else { cond_L = T_out(-1.0); }
      L_i += cond_L * T_out(local_model.gh_weights[k] * 0.5641895835477563);
    }
    out[i] = L_i;
    
    if (left_trunc) {
      double cum_haz = 0.0;
      for (int k = 0; k < n_inc_years; k++) {
        double age_at_year = inc_years[k] - data[i].dob;
        if (age_at_year < 0) continue;
        int age_idx = 0;
        if (age_at_year < 40.0) age_idx = 0;
        else if (age_at_year >= 85.0) age_idx = 10;
        else age_idx = (int)((age_at_year - 40.0) / 5.0) + 1;
        cum_haz += inc_rates[k][age_idx];
      }
      
      double X_Y_1997_2006 = std::exp(-cum_haz);
      double date_1997_days = 9862.0;
      double age_1997 = (date_1997_days - data[i].dob) / 365.25;
      T_out X_Y_0_1997(1.0);
      
      if (age_1997 > 0) {
        auto fn = [&](T_out x_ad) -> T_out {
          double x = as_double(x_ad);
          return local_model.f1(x) * local_model.S2(age_1997 - x);
        };
        X_Y_0_1997 = local_model.S1(age_1997) + boost::math::quadrature::gauss_kronrod<T_out, 15>::integrate(fn, T_out(0.0), T_out(age_1997), 5, T_out(local_model.tol), &local_model.error);
      }
      out[i] = out[i] / X_Y_0_1997 / T_out(X_Y_1997_2006);
    }
    if (weighted_ll) {
      using std::log;
      out[i] = T_out(weights[i]) * log(out[i]);
    }
  }
}
return out;
  }
};
} // end of namespace screening

#endif /* end of include guard: SCREENING_SCREENING_H */