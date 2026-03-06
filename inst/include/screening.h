#ifndef SCREENING_SCREENING_H
#define SCREENING_SCREENING_H = 1

// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <array>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <omp.h>

namespace screening {

// Helper to extract double value safely depending on whether T is double or cfaad::Number
inline double as_double(double x) { return x; }
template <class T>
inline double as_double(const T& x) { return x.value(); }

/**
 Currently, we assume four types of screening models:
 1. Screening episodes with end times of the episodes and whether a cancer was detected
 2. Screening episodes with end times of the episodes, the/a biomarker value and
 whether a cancer was detected
 3. Screening episodes with end times of the episodes, the/a biomarker value, whether a
 biopsy was undertaken and whether a cancer was detected
 4. Screening episodes with end times of the episodes, the/a biomarker value dependent on cancer onset, whether a
 biopsy was undertaken and whether a cancer was detected
 
 We have factored some functionality into `AbstractScreeningModel`.
 */

// Abstract templated class for a base screening model
// T_out dictates the output type (double or cfaad::Number)
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
      // Cast down to double for Rcpp output safely
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

// A simple screening model with onset and clinical diagnosis
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
    
    // AD Tape is thread-local, so disable OpenMP if T_out is not standard double
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


// Second screening model with onset, clinical diagnosis and test sensitivity
template<class T1, class T2, class T3, class T4, class T5>
class ScreeningModel2 : public AbstractScreeningModel<T1,T2,T3,T4, double> {
public:
  std::vector<double> yi;
  T5 PrFalseNeg;
  ScreeningModel2(T1 f1, T2 S1, T3 f2, T4 S2, T5 PrFalseNeg,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4, double>(f1,S1,f2,S2, tol),     
    PrFalseNeg(PrFalseNeg) {}
  void update(std::vector<double> ti,
              std::vector<double> yi) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti);
    this->yi=yi;
  }
  
  // overload for openmp
  void update(const double* ti_ptr, size_t ti_n, 
              const double* yi_ptr, size_t yi_n) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
  }
  
  double prod_beta(size_t i, size_t j, bool detected = false) {
    double value = 1.0;
    // reminder: yi[k] is not zero-padded => represents a test at tj[k+1]==ti[k]
    for (size_t k=i; k<j; k++) {
      value *= (detected && k+1==j ? 1.0-PrFalseNeg(yi[k]) : PrFalseNeg(yi[k]));
    }
    return value;
  }
  
  std::vector<double> likes(Rcpp::List inputs, double eps = 1.0e-12) {
    using Rcpp::as;
    size_t n_obs = inputs.size();
    std::vector<double> out(n_obs);
    
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
    
#pragma omp parallel 
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
        out[i] = local_model.Y(data[i].t-eps) * (1 - local_model.PrFalseNeg(local_model.yi[local_model.n-1]));
      } else {
        out[i] = 0.0; 
      }
    }
    else if (data[i].type == 3) { 
      out[i] = local_model.I(data[i].t);
    } 
    else {
      out[i] = -1.0;  
    }
  }
}
return out;
  }
};


// Third screening model with onset, clinical diagnosis and screening histories
// and test sensitivity
template<class T1, class T2, class T3, class T4, class T5>
class ScreeningModel3 : public AbstractScreeningModel<T1,T2,T3,T4, double> {
public:
  std::vector<double> yi; 
  std::vector<int> bxi; 
  T5 PrNoBx;
  double PrFalseNegBx;
  
  ScreeningModel3(T1 f1, T2 S1, T3 f2, T4 S2,
                  T5 PrNoBx,
                  double PrFalseNegBx,
                  double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4, double>(f1,S1,f2,S2,tol),
    PrNoBx(PrNoBx), PrFalseNegBx(PrFalseNegBx) { }
  
  void update(std::vector<double> ti, std::vector<double> yi, std::vector<int> bxi) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti);
    this->yi=yi;
    this->bxi=bxi;
  }
  
  void update(const double* ti_ptr, size_t ti_n, 
              const double* yi_ptr, size_t yi_n, 
              const int* bxi_ptr, size_t bxi_n) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
    this->bxi.assign(bxi_ptr, bxi_ptr + bxi_n);
  }
  
  double prod_bx(size_t i, size_t j) {
    double value = 1.0;
    for (size_t k=i; k<j; k++)
      value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (1.0-PrNoBx(yi[k])));
    return value;
  }
  
  double prod_beta(size_t i, size_t j, bool detected = false) {
    double value = 1.0;
    for (size_t k=i; k<j; k++)
      value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (1.0-PrNoBx(yi[k]))*(detected && k+1==j ? (1.0 - PrFalseNegBx) : PrFalseNegBx));
    return value;
  }
  
  double like_neg_screening(double s, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(s);
    double value = this->S1(s)*prod_bx(0,this->n);
    for (size_t i=0; i<=this->n; i++) {
      double K = prod_bx(0,i) * prod_beta(i,this->n);
      auto fn = [&](double x) {
        return this->f1(x)*this->S2(s-x) * K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1],
                                                    5, this->tol, &this->error);
    }
    return value;
  }
  
  double like_screen_detected_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    double value = 0.0;
    for (size_t i=0; i<=this->n; i++) {
      double K = prod_bx(0,i) * prod_beta(i,this->n+this->offset,true);
      auto fn = [&](double x) {
        return this->f1(x)*this->S2(t-x)*K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
    }
    return value;
  }
  
  double like_interval_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    double value = 0.0;
    for (size_t i=0; i<=this->n; i++) {
      double K = prod_bx(0,i)*prod_beta(i,this->n+this->offset);
      auto fn = [&](double x) {
        return this->f1(x)*this->f2(t-x)*K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
    }
    return value;
  }
  
  std::vector<double> likes(Rcpp::List inputs, double eps=1.0e-12) override {
    return likes(inputs, eps, "", {}); 
  }
  
  std::vector<double> likes(Rcpp::List inputs, double eps=1.0e-12,
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
    std::vector<double> out(n_obs);
    
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
    
#pragma omp parallel 
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
      
      double X_Y_0_1997 = 1.0;
      if (age_1997 > 0) {
        auto fn = [&](double x) {
          return local_model.f1(x) * local_model.S2(age_1997 - x);
        };
        X_Y_0_1997 = local_model.S1(age_1997) + 
          boost::math::quadrature::gauss_kronrod<double, 15>::integrate(fn, 0.0, age_1997, 5,
                                                                        local_model.tol,
                                                                        &local_model.error);
      }
      out[i] = out[i] / X_Y_0_1997 / X_Y_1997_2006;
    }
    if(weighted_ll) out[i] = weights[i]*std::log(out[i]);
  }
}

return out;
  }
};


// ScreeningModel4: ScreeningModel3 + biomarker density is dependent on onset
template<class T1, class T2, class T3, class T4, class T5, class T6>
class ScreeningModel4 : public AbstractScreeningModel<T1,T2,T3,T4, double> {
public:
  std::vector<double> yi; 
  std::vector<int> bxi; 
  T5 PrNoBx;
  T6 biomarker_den;
  double PrFalseNegBx;
  
  ScreeningModel4(T1 f1, T2 S1, T3 f2, T4 S2, T5 PrNoBx, T6 biomarker_den, double PrFalseNegBx, double tol = 1e-6) :
    AbstractScreeningModel<T1,T2,T3,T4, double>(f1,S1,f2,S2,tol),
    PrNoBx(PrNoBx), biomarker_den(biomarker_den), PrFalseNegBx(PrFalseNegBx) { }
  
  void update(std::vector<double> ti, std::vector<double> yi, std::vector<int> bxi) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti);
    this->yi=yi;
    this->bxi=bxi;
  }
  
  void update(const double* ti_ptr, size_t ti_n, 
              const double* yi_ptr, size_t yi_n, 
              const int* bxi_ptr, size_t bxi_n) {
    AbstractScreeningModel<T1,T2,T3,T4, double>::update(ti_ptr, ti_n);
    this->yi.assign(yi_ptr, yi_ptr + yi_n);
    this->bxi.assign(bxi_ptr, bxi_ptr + bxi_n);
  }
  
  double prod_history(size_t i, size_t j, double x, bool detected = false) {
    double value = 1.0;
    for (size_t k=0; k<j; k++) {
      value *= biomarker_den(yi[k], this->tj[k+1], x);
      
      if (k < i) { 
        value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (1.0-PrNoBx(yi[k])));
      } else { 
        value *= (bxi[k]==0 ? PrNoBx(yi[k]) : (1.0-PrNoBx(yi[k]))*(detected && k+1==j ? (1.0 - PrFalseNegBx) : PrFalseNegBx));
      }
    }
    return value;
  }
  
  double prod_beta(size_t i, size_t j, bool detected = false) override {
    return 1.0; 
  }
  
  double like_neg_screening(double s, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(s);
    
    double no_onset_history = 1.0;
    for (size_t k=0; k<this->n; k++) {
      no_onset_history *= biomarker_den(yi[k], this->tj[k+1], 1e9); 
      no_onset_history *= (bxi[k]==0 ? PrNoBx(yi[k]) : (1.0-PrNoBx(yi[k])));
    }
    double value = this->S1(s) * no_onset_history;
    
    for (size_t i=0; i<=this->n; i++) {
      auto fn = [&](double x) {
        double K = prod_history(i, this->n, x, false);
        return this->f1(x)*this->S2(s-x) * K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1],
                                                    5, this->tol, &this->error);
    }
    return value;
  }
  
  double like_screen_detected_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    double value = 0.0;
    for (size_t i=0; i<=this->n; i++) {
      auto fn = [&](double x) {
        double K = prod_history(i, this->n+this->offset, x, true);
        return this->f1(x)*this->S2(t-x)*K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
    }
    return value;
  }
  
  double like_interval_cancer(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) this->setup(t);
    double value = 0.0;
    for (size_t i=0; i<=this->n; i++) {
      auto fn = [&](double x) {
        double K = prod_history(i, this->n+this->offset, x, false);
        return this->f1(x)*this->f2(t-x)*K;
      };
      value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
    }
    return value;
  }
  
  std::vector<double> likes(Rcpp::List inputs, double eps=1.0e-12) override {
    return likes(inputs, eps, "", {}); 
  }
  
  std::vector<double> likes(Rcpp::List inputs, double eps=1.0e-12,
                            std::string return_type = "", std::vector<double> weights = {},
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
    std::vector<double> out(n_obs);
    
    for(size_t i = 0; i < n_obs; ++i) {
      Rcpp::List subject = inputs[i];
      data[i].t = Rcpp::as<double>(subject["t"]);
      data[i].type = Rcpp::as<int>(subject["type"]);
      data[i].dob = Rcpp::as<double>(subject["dob"]);
      if(data[i].type >3) Rcpp::stop("Only 3 screen types are supported");
      data[i].ti = Rcpp::as<std::vector<double>>(subject["ti"]);
      data[i].yi = Rcpp::as<std::vector<double>>(subject["yi"]);
      data[i].bxi = Rcpp::as<std::vector<int>>(subject["bxi"]);
    }
    
#pragma omp parallel 
{
  using namespace boost::math::quadrature;
  ScreeningModel4 local_model = *this;
  
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
      double X_Y_0_1997 = 1.0;
      
      if (age_1997 > 0) {
        auto fn = [&](double x) {
          return local_model.f1(x) * local_model.S2(age_1997 - x);
        };
        X_Y_0_1997 = local_model.S1(age_1997) + 
          boost::math::quadrature::gauss_kronrod<double, 15>::integrate(fn, 0.0, age_1997, 5,
                                                                        local_model.tol, &local_model.error);
      }
      out[i] = out[i] / X_Y_0_1997 / X_Y_1997_2006;
    }
    if(weighted_ll) out[i] = weights[i]*std::log(out[i]);
  }
}
return out;
  }
};

} // end of namespace screening

#endif /* end of include guard: SCREENING_SCREENING_H */