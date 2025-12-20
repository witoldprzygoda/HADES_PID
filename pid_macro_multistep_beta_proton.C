// pid_macro_delta_beta.C
// PID analysis using Δβ = β_measured - β_pion(p) representation
// 
// IMPROVED FITTING STRATEGY:
// 1. Pre-fit: Signal Gauss + Polynomial (no background Gauss)
// 2. Main fit: Add background Gauss with constrained polynomial
// 3. Parameter propagation between neighboring slices

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TF1.h"
#include "TFitResultPtr.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TAxis.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TChain.h"
#include "TSystem.h"

using std::cout; using std::endl;

// =====================================================
// GLOBAL CONSTANTS
// =====================================================
const double gPionMass = 139.57;  // MeV/c²
const double gSSquared = 1e-4;    // For (mass, a) transformation

// =====================================================
// PHYSICS FUNCTIONS
// =====================================================

double betaPion(double p) {
  return p / std::sqrt(p * p + gPionMass * gPionMass);
}

double deltaBetaToBeta(double p, double dBeta) {
  return betaPion(p) + dBeta;
}

double pBetaToMass(double p, double beta) {
  if (beta <= 0 || beta >= 1.5) return -1;
  double mass2 = p * p * (1.0 / (beta * beta) - 1.0);
  return (mass2 >= 0) ? std::sqrt(mass2) : -std::sqrt(-mass2);
}

bool pBetaToMassA(double p, double beta, double s2, double& mass, double& a) {
  if (beta <= 0) return false;
  double mass2 = p * p * (1.0 / (beta * beta) - 1.0);
  mass = (mass2 >= 0) ? std::sqrt(mass2) : -std::sqrt(-mass2);
  
  double arg = 1.0 + s2 * p * p - std::pow(1.0 - beta * beta, 2);
  if (arg < 0) return false;
  a = std::sqrt(arg);
  return true;
}

// =====================================================
// FIT RESULT STRUCTURE
// =====================================================
struct FitResult {
  bool success;
  double mean;        // Δβ mean (should be near 0 for pions)
  double sigma;       // Δβ width
  double amplitude;   // Signal amplitude
  int polyOrder;
  double chi2ndf;
  double entries;
  double pCenter, pLow, pHigh, sliceWidth;
  int phaseTag;
  // Background Gauss parameters
  double bkgGausAmp, bkgGausMean, bkgGausSigma;
  // Polynomial parameters
  std::vector<double> polyParams;
};

// =====================================================
// PROPAGATED PARAMETERS STRUCTURE
// =====================================================
struct PropagatedParams {
  bool valid;
  double sigMean, sigSigma, sigAmp;
  double bkgAmp, bkgMean, bkgSigma;
  std::vector<double> polyParams;
  int polyOrder;
};

// =====================================================
// SMOOTHING FUNCTIONS
// =====================================================

std::vector<double> runningMedian(const std::vector<double>& data, int windowSize) {
  std::vector<double> result(data.size());
  int halfWin = windowSize / 2;
  
  for (size_t i = 0; i < data.size(); ++i) {
    std::vector<double> window;
    for (int j = -halfWin; j <= halfWin; ++j) {
      int idx = (int)i + j;
      if (idx >= 0 && idx < (int)data.size()) {
        window.push_back(data[idx]);
      }
    }
    std::sort(window.begin(), window.end());
    result[i] = window[window.size() / 2];
  }
  return result;
}

std::vector<double> gaussianSmooth(const std::vector<double>& data, int windowSize, double sigma) {
  std::vector<double> result(data.size());
  int halfWin = windowSize / 2;
  
  std::vector<double> kernel(windowSize);
  double sum = 0;
  for (int i = 0; i < windowSize; ++i) {
    double x = i - halfWin;
    kernel[i] = std::exp(-x*x / (2*sigma*sigma));
    sum += kernel[i];
  }
  for (int i = 0; i < windowSize; ++i) kernel[i] /= sum;
  
  for (size_t i = 0; i < data.size(); ++i) {
    double val = 0, wsum = 0;
    for (int j = -halfWin; j <= halfWin; ++j) {
      int idx = (int)i + j;
      if (idx >= 0 && idx < (int)data.size()) {
        double w = kernel[j + halfWin];
        val += w * data[idx];
        wsum += w;
      }
    }
    result[i] = val / wsum;
  }
  return result;
}

std::vector<double> robustSmooth(const std::vector<double>& data, int medianWin, int gaussWin) {
  std::vector<double> temp = runningMedian(data, medianWin);
  return gaussianSmooth(temp, gaussWin, gaussWin / 3.0);
}

// Sort three vectors together by the first vector (momentum)
void sortByMomentum(std::vector<double>& p, std::vector<double>& mean, std::vector<double>& sigma) {
  if (p.size() != mean.size() || p.size() != sigma.size()) return;
  
  // Create index vector
  std::vector<size_t> indices(p.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
  
  // Sort indices by momentum
  std::sort(indices.begin(), indices.end(), 
            [&p](size_t a, size_t b) { return p[a] < p[b]; });
  
  // Reorder all three vectors
  std::vector<double> pSorted(p.size()), meanSorted(p.size()), sigmaSorted(p.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    pSorted[i] = p[indices[i]];
    meanSorted[i] = mean[indices[i]];
    sigmaSorted[i] = sigma[indices[i]];
  }
  
  p = pSorted;
  mean = meanSorted;
  sigma = sigmaSorted;
}

// =====================================================
// IMPROVED FITTING FUNCTION
// Two-stage approach:
//   Stage 1: Signal Gauss + Polynomial (pol2, pol3, pol4) - MAIN FIT
//   Stage 2: Add TINY background Gauss (max 5% of signal) with FIXED polynomial
// =====================================================

FitResult tryFitDeltaBeta(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx,
                          double pCenter, double pLow, double pHigh, double sliceWidth, int phaseTag,
                          const PropagatedParams& prevParams) {
  FitResult best;
  best.success = false;
  best.mean = 0.0;
  best.sigma = 0.02;
  best.amplitude = 0.0;
  best.polyOrder = 2;
  best.chi2ndf = 1e9;
  best.entries = proj ? proj->GetEntries() : 0;
  best.pCenter = pCenter;
  best.pLow = pLow;
  best.pHigh = pHigh;
  best.sliceWidth = sliceWidth;
  best.phaseTag = phaseTag;
  best.bkgGausAmp = 0;
  best.bkgGausMean = 0;
  best.bkgGausSigma = 0.1;

  if (!proj || best.entries < 30) return best;

  int bin_min = proj->FindFixBin(fitMin);
  int bin_max = proj->FindFixBin(fitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 10.0) return best;

  // =====================================================
  // Find peak position in signal region near Δβ = 0
  // =====================================================
  int bin_sig_lo = proj->FindFixBin(-0.05);
  int bin_sig_hi = proj->FindFixBin(0.05);
  
  double peakVal = -1.0;
  int peakBin = 0;
  for (int b = bin_sig_lo; b <= bin_sig_hi; ++b) {
    double v = 0;
    int cnt = 0;
    for (int k = -1; k <= 1; ++k) {
      if (b + k >= 1 && b + k <= proj->GetNbinsX()) {
        v += proj->GetBinContent(b + k);
        cnt++;
      }
    }
    v /= cnt;
    if (v > peakVal) { 
      peakVal = v; 
      peakBin = b; 
    }
  }
  double peakPos = (peakBin > 0) ? proj->GetBinCenter(peakBin) : 0.0;
  peakVal = (peakBin > 0) ? proj->GetBinContent(peakBin) : 1.0;

  // Background estimate from edges
  double bkgLeft = 0, bkgRight = 0;
  int nEdgeBins = 5;
  for (int b = bin_min; b < bin_min + nEdgeBins && b <= bin_max; ++b)
    bkgLeft += proj->GetBinContent(b);
  for (int b = bin_max; b > bin_max - nEdgeBins && b >= bin_min; --b)
    bkgRight += proj->GetBinContent(b);
  bkgLeft /= nEdgeBins;
  bkgRight /= nEdgeBins;
  double bkgAvg = 0.5 * (bkgLeft + bkgRight);

  double sigAmpEst = std::max(1.0, peakVal - bkgAvg);

  // =====================================================
  // Use propagated parameters if available
  // =====================================================
  double initSigMean = peakPos;
  double initSigSigma = 0.015;
  double initSigAmp = sigAmpEst;
  
  if (prevParams.valid) {
    // ALWAYS use propagated signal params - they are reliable
    initSigMean = prevParams.sigMean;
    initSigSigma = prevParams.sigSigma;
    // Scale amplitude by relative statistics, but keep it reasonable
    double ampScale = integral / 1000.0;
    initSigAmp = prevParams.sigAmp * ampScale;
    // Ensure amplitude is reasonable
    if (initSigAmp < 0.5) initSigAmp = sigAmpEst;
    if (initSigAmp > 10.0 * peakVal) initSigAmp = sigAmpEst;
    
    // Debug: show that we're using propagated params
    if (sliceIdx < 5 || phaseTag == 0) {  // Show for Phase 0
      //cout << Form("    [DEBUG] Using propagated: μ=%.4f, σ=%.4f, amp=%.1f (scaled)", 
      //             initSigMean, initSigSigma, initSigAmp) << endl;
    }
  }

  // =====================================================
  // Try polynomial orders 2, 3, 4
  // =====================================================
  struct StageResult {
    bool valid;
    double chi2ndf_stage1;
    double chi2ndf;
    double sigMean, sigSigma, sigAmp;
    double bkgAmp, bkgMean, bkgSigma;
    std::vector<double> polyParams;
    int polyOrder;
  };
  
  const int polyOrders[3] = {2, 3, 4};
  const int nModels = 3;
  StageResult results[nModels];
  for (int i = 0; i < nModels; ++i) results[i].valid = false;

  for (int m = 0; m < nModels; ++m) {
    int polyOrder = polyOrders[m];
    int nPolyParams = polyOrder + 1;
    
    // =====================================================
    // STAGE 1: Signal Gaussian + Polynomial ONLY
    // This is the MAIN fit - polynomial handles ALL background
    // =====================================================
    TString funcExpr1 = "gaus(0)";
    for (int pp = 0; pp <= polyOrder; ++pp) {
      if (pp == 0) funcExpr1 += Form(" + [%d]", 3 + pp);
      else funcExpr1 += Form(" + [%d]*pow(x,%d)", 3 + pp, pp);
    }
    
    TString funcName1 = Form("stage1_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* stage1Fit = new TF1(funcName1, funcExpr1, fitMin, fitMax);
    
    // Signal Gaussian
    stage1Fit->SetParameter(0, initSigAmp);
    stage1Fit->SetParameter(1, initSigMean);
    stage1Fit->SetParameter(2, initSigSigma);
    stage1Fit->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal));
    stage1Fit->SetParLimits(1, -0.04, 0.04);
    stage1Fit->SetParLimits(2, 0.003, 0.06);
    
    // Polynomial initialization - ALWAYS use propagated params as base
    if (prevParams.valid && prevParams.polyParams.size() > 0) {
      // Use propagated polynomial params as starting point
      // Copy what we can from previous fit
      stage1Fit->SetParameter(3, prevParams.polyParams[0]);  // constant
      if (polyOrder >= 1 && prevParams.polyParams.size() > 1) 
        stage1Fit->SetParameter(4, prevParams.polyParams[1]);
      else if (polyOrder >= 1)
        stage1Fit->SetParameter(4, 0.0);
      if (polyOrder >= 2 && prevParams.polyParams.size() > 2) 
        stage1Fit->SetParameter(5, prevParams.polyParams[2]);
      else if (polyOrder >= 2)
        stage1Fit->SetParameter(5, 0.0);
      if (polyOrder >= 3 && prevParams.polyParams.size() > 3) 
        stage1Fit->SetParameter(6, prevParams.polyParams[3]);
      else if (polyOrder >= 3)
        stage1Fit->SetParameter(6, 0.0);
      if (polyOrder >= 4 && prevParams.polyParams.size() > 4) 
        stage1Fit->SetParameter(7, prevParams.polyParams[4]);
      else if (polyOrder >= 4)
        stage1Fit->SetParameter(7, 0.0);
    } else {
      // No propagated params - use defaults
      stage1Fit->SetParameter(3, std::max(0.1, bkgAvg));
      double slope = (bkgRight - bkgLeft) / (fitMax - fitMin);
      if (polyOrder >= 1) stage1Fit->SetParameter(4, slope);
      if (polyOrder >= 2) stage1Fit->SetParameter(5, 0.0);
      if (polyOrder >= 3) stage1Fit->SetParameter(6, 0.0);
      if (polyOrder >= 4) stage1Fit->SetParameter(7, 0.0);
    }

    TFitResultPtr res1 = proj->Fit(stage1Fit, "SQR0B");
    
    if (!res1.Get() || !res1->IsValid() || res1->Status() != 0) {
      delete stage1Fit;
      continue;
    }
    
    // Extract Stage 1 results
    double s1_sigAmp = stage1Fit->GetParameter(0);
    double s1_sigMean = stage1Fit->GetParameter(1);
    double s1_sigSigma = stage1Fit->GetParameter(2);
    std::vector<double> s1_poly;
    for (int pp = 0; pp < nPolyParams; ++pp)
      s1_poly.push_back(stage1Fit->GetParameter(3 + pp));
    
    double s1_chi2ndf = (res1->Ndf() > 0) ? res1->Chi2() / res1->Ndf() : 1e6;
    
    delete stage1Fit;
    
    // Skip if Stage 1 fit is terrible
    if (s1_chi2ndf > 15.0) {
      if (phaseTag == 0) cout << Form("      [pol%d Stage1 chi2=%.1f > 15]", polyOrder, s1_chi2ndf) << endl;
      continue;
    }
    
    // Basic validation of Stage 1 - relaxed for wider windows
    double s1MeanRange = (sliceWidth > 50) ? 0.06 : 0.05;
    double s1AlignTol = (sliceWidth > 50) ? 0.04 : 0.03;
    
    if (s1_sigMean < -s1MeanRange || s1_sigMean > s1MeanRange) {
      if (phaseTag == 0) cout << Form("      [pol%d Stage1 mean=%.4f out of range ±%.2f]", polyOrder, s1_sigMean, s1MeanRange) << endl;
      continue;
    }
    if (s1_sigSigma < 0.003 || s1_sigSigma > 0.08) {
      if (phaseTag == 0) cout << Form("      [pol%d Stage1 sigma=%.4f out of range]", polyOrder, s1_sigSigma) << endl;
      continue;
    }
    if (std::abs(s1_sigMean - peakPos) > s1AlignTol) {
      if (phaseTag == 0) cout << Form("      [pol%d Stage1 peakAlign |%.4f-%.4f|=%.4f > %.3f]", 
                                       polyOrder, s1_sigMean, peakPos, std::abs(s1_sigMean - peakPos), s1AlignTol) << endl;
      continue;
    }

    // =====================================================
    // STAGE 2: Add TINY Background Gaussian
    // - Polynomial parameters are COMPLETELY FIXED
    // - Background Gauss amplitude limited to 5% of signal
    // - This is just fine-tuning, should barely change anything
    // =====================================================
    
    // Build expression: gaus(0) + gaus(3) + poly starting at [6]
    TString funcExpr2 = "gaus(0) + gaus(3)";
    for (int pp = 0; pp <= polyOrder; ++pp) {
      if (pp == 0) funcExpr2 += Form(" + [%d]", 6 + pp);
      else funcExpr2 += Form(" + [%d]*pow(x,%d)", 6 + pp, pp);
    }
    
    TString funcName2 = Form("stage2_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* stage2Fit = new TF1(funcName2, funcExpr2, fitMin, fitMax);
    
    // Signal Gaussian - very tight constraints around Stage 1
    stage2Fit->SetParameter(0, s1_sigAmp);
    stage2Fit->SetParameter(1, s1_sigMean);
    stage2Fit->SetParameter(2, s1_sigSigma);
    // Allow only ±10% variation
    stage2Fit->SetParLimits(0, s1_sigAmp * 0.9, s1_sigAmp * 1.1);
    stage2Fit->SetParLimits(1, s1_sigMean - 0.005, s1_sigMean + 0.005);
    stage2Fit->SetParLimits(2, s1_sigSigma * 0.9, s1_sigSigma * 1.1);
    
    // Background Gaussian - EXTREMELY SMALL correction
    // Maximum amplitude: 5% of signal amplitude
    double maxBkgAmp = 0.05 * s1_sigAmp;
    stage2Fit->SetParameter(3, 0.02 * s1_sigAmp);  // Start at 2%
    stage2Fit->SetParLimits(3, 0.0, maxBkgAmp);
    
    // Background Gaussian mean - away from signal
    stage2Fit->SetParameter(4, 0.05);
    stage2Fit->SetParLimits(4, 0.02, 0.10);
    
    // Background Gaussian sigma - broad
    stage2Fit->SetParameter(5, 0.06);
    stage2Fit->SetParLimits(5, 0.04, 0.10);
    
    // Polynomial - COMPLETELY FIXED from Stage 1
    for (int pp = 0; pp < nPolyParams; ++pp) {
      stage2Fit->SetParameter(6 + pp, s1_poly[pp]);
      stage2Fit->FixParameter(6 + pp, s1_poly[pp]);  // FIXED!
    }

    TFitResultPtr res2 = proj->Fit(stage2Fit, "SQR0B");
    
    bool stage2OK = false;
    double s2_sigAmp = s1_sigAmp;
    double s2_sigMean = s1_sigMean;
    double s2_sigSigma = s1_sigSigma;
    double s2_bkgAmp = 0;
    double s2_bkgMean = 0.05;
    double s2_bkgSigma = 0.06;
    double s2_chi2ndf = s1_chi2ndf;
    
    if (res2.Get() && res2->IsValid() && res2->Status() == 0) {
      s2_sigAmp = stage2Fit->GetParameter(0);
      s2_sigMean = stage2Fit->GetParameter(1);
      s2_sigSigma = stage2Fit->GetParameter(2);
      s2_bkgAmp = stage2Fit->GetParameter(3);
      s2_bkgMean = stage2Fit->GetParameter(4);
      s2_bkgSigma = stage2Fit->GetParameter(5);
      int ndf = res2->Ndf();
      double chi2 = res2->Chi2();
      s2_chi2ndf = (ndf > 0) ? chi2 / ndf : s1_chi2ndf;
      
      // Check if Stage 2 improved the fit (or at least didn't make it worse)
      if (s2_chi2ndf <= s1_chi2ndf * 1.1) {
        stage2OK = true;
      }
    }
    
    delete stage2Fit;
    
    // If Stage 2 failed or made things worse, use Stage 1 results with bkgAmp=0
    if (!stage2OK) {
      s2_sigAmp = s1_sigAmp;
      s2_sigMean = s1_sigMean;
      s2_sigSigma = s1_sigSigma;
      s2_bkgAmp = 0;
      s2_chi2ndf = s1_chi2ndf;
    }
    
    // Final validation - relaxed for wider windows
    bool sigmaOK = (s2_sigSigma > 0.003 && s2_sigSigma < 0.08);  // Wider sigma allowed
    bool meanOK = (s2_sigMean > -0.05 && s2_sigMean < 0.05);     // Slightly wider mean range
    bool ampOK = (s2_sigAmp > 0.3);                               // Lower amp threshold
    bool chi2OK = (s2_chi2ndf > 0.02 && s2_chi2ndf < 30.0);      // Wider chi2 range
    // Relax peakAligned for wider windows (Phase 0 extended slices)
    double alignTol = 0.03;  // Base tolerance
    if (sliceWidth > 50) alignTol = 0.05;  // More tolerance for wide windows
    bool peakAligned = (std::abs(s2_sigMean - peakPos) < alignTol);

    // Debug output for Phase 0 failures
    if (phaseTag == 0 && !(sigmaOK && meanOK && ampOK && chi2OK && peakAligned)) {
      cout << Form("      [pol%d FAIL] μ=%.4f(OK:%d), σ=%.4f(OK:%d), A=%.1f(OK:%d), χ²=%.2f(OK:%d), align=%.3f(OK:%d)",
                   polyOrder, s2_sigMean, meanOK, s2_sigSigma, sigmaOK, s2_sigAmp, ampOK, 
                   s2_chi2ndf, chi2OK, std::abs(s2_sigMean - peakPos), peakAligned) << endl;
    }

    if (sigmaOK && meanOK && ampOK && chi2OK && peakAligned) {
      results[m].valid = true;
      results[m].chi2ndf_stage1 = s1_chi2ndf;
      results[m].chi2ndf = s2_chi2ndf;
      results[m].sigMean = s2_sigMean;
      results[m].sigSigma = s2_sigSigma;
      results[m].sigAmp = s2_sigAmp;
      results[m].bkgAmp = s2_bkgAmp;
      results[m].bkgMean = s2_bkgMean;
      results[m].bkgSigma = s2_bkgSigma;
      results[m].polyOrder = polyOrder;
      results[m].polyParams = s1_poly;  // Always use Stage 1 polynomial
    }
  }

  // =====================================================
  // Select best model - prefer simpler polynomial if chi2 OK
  // =====================================================
  const double chi2_good_min = 0.3;
  const double chi2_good_max = 3.0;
  
  int bestIdx = -1;
  double bestScore = 1e9;
  
  // First: find simplest model with good chi2
  for (int m = 0; m < nModels; ++m) {
    if (results[m].valid) {
      double c = results[m].chi2ndf;
      if (c >= chi2_good_min && c <= chi2_good_max) {
        bestIdx = m;
        break;
      }
    }
  }
  
  // Fallback: use scoring
  if (bestIdx < 0) {
    for (int m = 0; m < nModels; ++m) {
      if (results[m].valid) {
        double c = results[m].chi2ndf;
        double score = std::abs(std::log(c)) + 0.1 * (polyOrders[m] - 2);
        if (score < bestScore) {
          bestScore = score;
          bestIdx = m;
        }
      }
    }
  }
  
  if (bestIdx >= 0) {
    best.success = true;
    best.mean = results[bestIdx].sigMean;
    best.sigma = results[bestIdx].sigSigma;
    best.amplitude = results[bestIdx].sigAmp;
    best.polyOrder = results[bestIdx].polyOrder;
    best.chi2ndf = results[bestIdx].chi2ndf;
    best.bkgGausAmp = results[bestIdx].bkgAmp;
    best.bkgGausMean = results[bestIdx].bkgMean;
    best.bkgGausSigma = results[bestIdx].bkgSigma;
    best.polyParams = results[bestIdx].polyParams;
  }

  return best;
}

// Convert FitResult to PropagatedParams
PropagatedParams fitResultToProps(const FitResult& r) {
  PropagatedParams p;
  p.valid = r.success;
  if (r.success) {
    p.sigMean = r.mean;
    p.sigSigma = r.sigma;
    p.sigAmp = r.amplitude;
    p.bkgAmp = r.bkgGausAmp;
    p.bkgMean = r.bkgGausMean;
    p.bkgSigma = r.bkgGausSigma;
    p.polyParams = r.polyParams;
    p.polyOrder = r.polyOrder;
  }
  return p;
}

// =====================================================
// MAIN FUNCTION
// =====================================================

void pid_macro_multistep_beta_proton() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain
  const char* treeName = "PEpEm_ID";
  const std::vector<TString> files = {
    "pp060_01_exp.root","pp060_02_exp.root","pp060_03_exp.root",
    "pp060_04_exp.root","pp060_05_exp.root","pp060_06_exp.root",
    "pp060_07_exp.root","pp060_08_exp.root","pp060_09_exp.root", "pp060_10_exp.root"
  };
  TChain* chain = new TChain(treeName);
  int added = 0;
  for (const auto& fn : files) {
    if (gSystem->AccessPathName(fn)) {
      cout << "[WARN] missing file: " << fn << endl;
      continue;
    }
    chain->Add(fn);
    ++added;
  }
  Long64_t nEnt = chain->GetEntries();
  if (added == 0 || nEnt <= 0) {
    cout << "No entries found." << endl;
    return;
  }
  cout << "TChain: " << added << " files, " << nEnt << " entries" << endl;

  // =====================================================
  // CREATE 2D HISTOGRAMS
  // =====================================================
  
  // Main histogram: p vs Δβ
  const char* h2name = "h2_p_deltaBeta";
  TString drawCmd = Form(
    "p_beta - p_p/sqrt(p_p*p_p + %.2f*%.2f) : p_p >> %s(280,0,1400,300,-0.15,0.15)",
    gPionMass, gPionMass, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "isBest==1", "colz");
  TH2F* h2DB = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2DB) {
    cout << "Failed to create histogram" << endl;
    return;
  }

  h2DB->SetTitle("Momentum vs #Delta#beta (#beta - #beta_{#pi})");
  h2DB->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DB->GetYaxis()->SetTitle("#Delta#beta = #beta - #beta_{#pi}");

  // (p, β) histogram
  const char* h2pb_name = "h2_p_beta";
  if (gDirectory->FindObject(h2pb_name)) gDirectory->Delete(Form("%s;*", h2pb_name));
  chain->Draw(Form("p_beta : p_p >> %s(280,0,1400,300,0.3,1.15)", h2pb_name), "isBest==1", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2pb_name));

  // (mass, a) histogram
  const char* h2ma_name = "h2_mass_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*p_p*p_p - pow(1-p_beta*p_beta,2)) : "
    "p_p*sqrt(1/pow(p_beta,2) - 1) >> %s(300,0,300,160,0,16)",
    gSSquared, h2ma_name);
  if (gDirectory->FindObject(h2ma_name)) gDirectory->Delete(Form("%s;*", h2ma_name));
  chain->Draw(drawMA, "isBest==1 && p_beta>0 && p_beta<1", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2ma_name));

  // (p, mass²) histogram - mass² = p² * (1/β² - 1)
  const char* h2m2_name = "h2_p_mass2";
  if (gDirectory->FindObject(h2m2_name)) gDirectory->Delete(Form("%s;*", h2m2_name));
  chain->Draw(Form("p_p*p_p*(1.0/(p_beta*p_beta) - 1) : p_p >> %s(280,0,1400,400,-20000,60000)", h2m2_name), 
              "isBest==1 && p_beta>0.1 && p_beta<1.5", "colz");
  TH2F* h2M2 = static_cast<TH2F*>(gDirectory->Get(h2m2_name));
  if (h2M2) {
    h2M2->SetTitle("Mass^{2} vs Momentum");
    h2M2->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2M2->GetYaxis()->SetTitle("Mass^{2} [MeV^{2}/c^{4}]");
  }

  // =====================================================
  // SCANNING PARAMETERS
  // =====================================================
  
  const double warmupLow = 200.0;       // Warmup fit range [200, 400] - always good
  const double warmupHigh = 400.0;
  const double startMom = 180.0;        // Anchor point for regular fitting
  const double transitionMom = 500.0;   // Where doubling starts
  const double endMom = 1400.0;
  const double stepSize = 1.0;          // 1 MeV/c steps in Phase 1
  
  const int nWidths = 4;
  const double baseWidths[nWidths] = {5.0, 10.0, 20.0, 40.0};
  const TString widthLabels[nWidths] = {"1x(5)", "2x(10)", "4x(20)", "8x(40)"};
  const int widthColors[nWidths] = {kBlue, kRed, kGreen+2, kMagenta};

  const double fitRangeMin = -0.12;
  const double fitRangeMax = 0.12;

  std::vector<std::vector<FitResult>> allFitResults(nWidths);
  std::vector<std::vector<double>> allMomCenters(nWidths);

  TCanvas* cFits[4];  // 4 widths now
  const int nDisplayPerWidth = 12;

  // =====================================================
  // PROCESS EACH WIDTH
  // =====================================================
  
  for (int w = 0; w < nWidths; ++w) {
    double baseW = baseWidths[w];
    cout << "\n=============================================" << endl;
    cout << "=== Width " << widthLabels[w] << " (base=" << baseW << " MeV/c) ===" << endl;
    cout << "=============================================" << endl;

    // =====================================================
    // BUILD ALL SLICES
    // Phase 0: backward - RIGHT EDGE ANCHORED at 180, extend left
    //          [175, 180], [170, 180], ..., [0, 180]
    // Phase 1: forward from startMom with fixed width (sliding)
    // Phase 2: forward from transitionMom with doubling width
    // =====================================================
    
    std::vector<double> momLows, momHighs, momCenters, sliceWidths;
    std::vector<int> phaseTag;

    // Phase 0: Right edge anchored at startMom, extend left edge
    // First slice: [180-baseW, 180], then [180-2*baseW, 180], etc.
    std::vector<int> phase0Indices;
    {
      double pRight = startMom;  // Always 180
      double pLeft = startMom - baseW;
      
      while (pLeft >= 0) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);  // Always startMom (180)
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(0);
        phase0Indices.push_back(idx);
        
        // Extend left edge by baseW for next slice (width grows)
        pLeft -= baseW;
      }
      
      // Final slice from 0 to startMom if not already there
      if (momLows.empty() || momLows.back() > 0) {
        int idx = momLows.size();
        momLows.push_back(0);
        momHighs.push_back(pRight);  // 180
        momCenters.push_back(0.5 * pRight);
        sliceWidths.push_back(pRight);
        phaseTag.push_back(0);
        phase0Indices.push_back(idx);
      }
    }
    int nPhase0 = phase0Indices.size();

    // Phase 1: forward sliding slices from startMom
    std::vector<int> phase1Indices;
    {
      double pLeft = startMom;
      while (pLeft < transitionMom) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pLeft + baseW);
        momCenters.push_back(pLeft + 0.5 * baseW);
        sliceWidths.push_back(baseW);
        phaseTag.push_back(1);
        phase1Indices.push_back(idx);
        pLeft += stepSize;
      }
    }
    int nPhase1 = phase1Indices.size();

    // Phase 2: doubling width from transitionMom
    std::vector<int> phase2Indices;
    {
      double pLeft = transitionMom;
      double currentWidth = baseW;
      while (pLeft < endMom) {
        double pRight = std::min(pLeft + currentWidth, endMom);
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(2);
        phase2Indices.push_back(idx);
        pLeft = pRight;
        currentWidth *= 2.0;
      }
    }
    int nPhase2 = phase2Indices.size();

    int nSlices = momLows.size();
    
    // Print Phase 0 slice structure
    cout << "Phase 0 slices (right edge anchored at " << startMom << "):" << endl;
    for (size_t k = 0; k < phase0Indices.size(); ++k) {
      int idx = phase0Indices[k];
      if (k < 5 || k >= phase0Indices.size() - 2) {
        cout << Form("  [%3.0f, %3.0f] width=%3.0f", 
                     momLows[idx], momHighs[idx], sliceWidths[idx]) << endl;
      } else if (k == 5) {
        cout << "  ..." << endl;
      }
    }
    
    cout << "Slices: Phase0=" << nPhase0 << " (backward, anchored), Phase1=" << nPhase1 
         << " (forward), Phase2=" << nPhase2 << " (doubling), Total=" << nSlices << endl;

    allMomCenters[w] = momCenters;
    allFitResults[w].resize(nSlices);

    // =====================================================
    // FIT IN CORRECT ORDER:
    // 0. WARMUP: [200, 400] - fixed range, always good, get initial params
    // 1. First slice of Phase 1 [180, 180+baseW] - use warmup params
    // 2. Phase 0: use anchor params, fit [175,180], [170,180], ..., [0,180]
    // 3. Rest of Phase 1 forward
    // 4. Phase 2 forward
    // =====================================================
    
    int nSuccess = 0;
    PropagatedParams warmupParams;
    warmupParams.valid = false;
    PropagatedParams anchorParams;
    anchorParams.valid = false;
    
    // Step 0: WARMUP FIT [180, 250] - this always works well!
    cout << "Fitting WARMUP [" << warmupLow << "," << warmupHigh << "] (fixed range, always good)..." << endl;
    {
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(warmupLow + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(warmupHigh - 0.01);
      
      TString projName = Form("proj_db_w%d_warmup", w);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      
      PropagatedParams noParams;
      noParams.valid = false;
      
      // Use special warmup parameters - this is a wide, high-statistics slice
      double warmupCenter = 0.5 * (warmupLow + warmupHigh);
      double warmupWidth = warmupHigh - warmupLow;
      
      FitResult warmupResult = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, -1, w,
                                                warmupCenter, warmupLow, warmupHigh, 
                                                warmupWidth, -1, noParams);  // phaseTag=-1 for warmup
      delete proj;
      
      if (warmupResult.success) {
        warmupParams = fitResultToProps(warmupResult);
        cout << Form("  WARMUP OK: μ=%.4f, σ=%.4f, χ²=%.2f (will use for all fits)", 
                     warmupResult.mean, warmupResult.sigma, warmupResult.chi2ndf) << endl;
      } else {
        cout << "  WARMUP FAILED! Will try fitting without initial params." << endl;
      }
    }
    
    // Step 1: Fit ANCHOR from Phase 1 [180, 180+baseW] using warmup params
    if (!phase1Indices.empty()) {
      int anchorIdx = phase1Indices[0];
      cout << "Fitting ANCHOR [" << momLows[anchorIdx] << "," << momHighs[anchorIdx] 
           << "] using warmup params..." << endl;
      
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[anchorIdx] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[anchorIdx] - 0.01);
      
      TString projName = Form("proj_db_w%d_anchor", w);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      
      // Use warmup params as starting point
      allFitResults[w][anchorIdx] = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, anchorIdx, w,
                                                     momCenters[anchorIdx], momLows[anchorIdx], 
                                                     momHighs[anchorIdx], sliceWidths[anchorIdx], 
                                                     phaseTag[anchorIdx], warmupParams);
      delete proj;
      
      if (allFitResults[w][anchorIdx].success) {
        nSuccess++;
        anchorParams = fitResultToProps(allFitResults[w][anchorIdx]);
        cout << Form("  ANCHOR OK: μ=%.4f, σ=%.4f, χ²=%.2f", 
                     allFitResults[w][anchorIdx].mean,
                     allFitResults[w][anchorIdx].sigma,
                     allFitResults[w][anchorIdx].chi2ndf) << endl;
      } else {
        cout << "  ANCHOR FAILED! This is unexpected." << endl;
      }
    }
    
    // Step 2: Fit Phase 0 using anchor parameters
    // Slices: [175,180], [170,180], ..., [0,180] - all with right edge at 180
    cout << "Fitting Phase 0 (backward, using anchor params)..." << endl;
    PropagatedParams prevParams = anchorParams;  // START WITH ANCHOR PARAMS!
    
    for (size_t k = 0; k < phase0Indices.size(); ++k) {
      int i = phase0Indices[k];
      
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[i] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[i] - 0.01);
      
      TString projName = Form("proj_db_w%d_s%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");

      allFitResults[w][i] = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, i, w,
                                             momCenters[i], momLows[i], momHighs[i], 
                                             sliceWidths[i], phaseTag[i], prevParams);
      
      if (allFitResults[w][i].success) {
        nSuccess++;
        prevParams = fitResultToProps(allFitResults[w][i]);
        
        if (k < 3 || k == phase0Indices.size() - 1) {
          cout << Form("  P0[%2zu] [%3.0f,%3.0f] w=%3.0f: μ=%7.4f, σ=%6.4f, χ²=%.2f",
                       k, momLows[i], momHighs[i], sliceWidths[i],
                       allFitResults[w][i].mean, allFitResults[w][i].sigma,
                       allFitResults[w][i].chi2ndf) << endl;
        } else if (k == 3) {
          cout << "  ..." << endl;
        }
      } else {
        // DON'T update prevParams on failure - keep using last good params
        cout << Form("  P0[%2zu] [%3.0f,%3.0f] w=%3.0f: FAILED (entries=%.0f, prevParams.valid=%d)",
                     k, momLows[i], momHighs[i], sliceWidths[i], 
                     allFitResults[w][i].entries, prevParams.valid ? 1 : 0) << endl;
      }
      // Keep previous params even if this fit failed
      
      delete proj;
    }
    
    // Print Phase 0 summary
    int p0Success = 0;
    for (size_t k = 0; k < phase0Indices.size(); ++k) {
      int i = phase0Indices[k];
      if (allFitResults[w][i].success) p0Success++;
    }
    cout << "Phase 0 success: " << p0Success << "/" << nPhase0 << endl;
    
    // Step 3: Fit rest of Phase 1 FORWARD (skip anchor which is already done)
    cout << "Fitting Phase 1 (forward from " << startMom << " to " << transitionMom << ")..." << endl;
    prevParams = anchorParams;  // Restart from anchor
    
    for (size_t k = 1; k < phase1Indices.size(); ++k) {  // Start from 1, skip anchor
      int i = phase1Indices[k];
      
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[i] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[i] - 0.01);
      
      TString projName = Form("proj_db_w%d_s%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");

      allFitResults[w][i] = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, i, w,
                                             momCenters[i], momLows[i], momHighs[i], 
                                             sliceWidths[i], phaseTag[i], prevParams);
      
      if (allFitResults[w][i].success) {
        nSuccess++;
        prevParams = fitResultToProps(allFitResults[w][i]);
      }
      
      delete proj;
    }
    
    // Get params from end of Phase 1 for Phase 2
    PropagatedParams phase1EndParams = prevParams;
    
    // Step 4: Fit Phase 2 FORWARD
    cout << "Fitting Phase 2 (doubling from " << transitionMom << " to " << endMom << ")..." << endl;
    prevParams = phase1EndParams;
    
    for (size_t k = 0; k < phase2Indices.size(); ++k) {
      int i = phase2Indices[k];
      
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[i] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[i] - 0.01);
      
      TString projName = Form("proj_db_w%d_s%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");

      allFitResults[w][i] = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, i, w,
                                             momCenters[i], momLows[i], momHighs[i], 
                                             sliceWidths[i], phaseTag[i], prevParams);
      
      if (allFitResults[w][i].success) {
        nSuccess++;
        prevParams = fitResultToProps(allFitResults[w][i]);
        cout << Form("  P2[%zu] p=[%6.0f,%6.0f] w=%4.0f: μ=%7.4f, σ=%6.4f, χ²=%.2f",
                     k, momLows[i], momHighs[i], sliceWidths[i],
                     allFitResults[w][i].mean, allFitResults[w][i].sigma,
                     allFitResults[w][i].chi2ndf) << endl;
      } else {
        cout << Form("  P2[%zu] p=[%6.0f,%6.0f] w=%4.0f: FAIL", 
                     k, momLows[i], momHighs[i], sliceWidths[i]) << endl;
      }
      
      delete proj;
    }
    
    cout << "Total success: " << nSuccess << "/" << nSlices << endl;

    // =====================================================
    // SELECT DISPLAY INDICES (after all fits are done)
    // =====================================================
    std::vector<int> displayIndices;
    
    // From Phase 0: first (anchor), a few intermediate, last (0-120)
    if (nPhase0 > 0) {
      displayIndices.push_back(phase0Indices[0]);  // Anchor [120-baseW, 120]
      if (nPhase0 > 3) displayIndices.push_back(phase0Indices[nPhase0/3]);
      if (nPhase0 > 2) displayIndices.push_back(phase0Indices[2*nPhase0/3]);
      displayIndices.push_back(phase0Indices[nPhase0-1]);  // [0, 120]
    }
    
    // From Phase 1: a few representative
    if (nPhase1 > 0) {
      for (int k = 0; k <= 4 && k < nPhase1; ++k) {
        int idx = k * nPhase1 / 5;
        if (idx < nPhase1) displayIndices.push_back(phase1Indices[idx]);
      }
    }
    
    // From Phase 2: all (usually just a few)
    for (size_t k = 0; k < phase2Indices.size() && k < 4; ++k) {
      displayIndices.push_back(phase2Indices[k]);
    }
    
    // Remove duplicates and sort by momentum center
    std::sort(displayIndices.begin(), displayIndices.end(), 
              [&momCenters](int a, int b) { return momCenters[a] < momCenters[b]; });
    displayIndices.erase(std::unique(displayIndices.begin(), displayIndices.end()), displayIndices.end());
    if (displayIndices.size() > (size_t)nDisplayPerWidth)
      displayIndices.resize(nDisplayPerWidth);

    cFits[w] = new TCanvas(Form("c_fits_dBeta_%d", w), 
                           Form("Delta-Beta Fits - %s", widthLabels[w].Data()), 1400, 900);
    cFits[w]->Divide(4, 3);

    // =====================================================
    // DISPLAY FITS
    // =====================================================
    for (size_t d = 0; d < displayIndices.size() && d < (size_t)nDisplayPerWidth; ++d) {
      int i = displayIndices[d];
      FitResult& result = allFitResults[w][i];

      cFits[w]->cd(d + 1);
      gPad->SetLeftMargin(0.14);
      gPad->SetRightMargin(0.04);
      gPad->SetTopMargin(0.10);
      gPad->SetBottomMargin(0.12);

      int xbin_lo = h2DB->GetXaxis()->FindFixBin(result.pLow + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(result.pHigh - 0.01);
      
      TString projName = Form("proj_disp_db_w%d_d%d", w, (int)d);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      if (!proj) continue;

      TString phaseStr = (result.phaseTag == 0) ? " [BWD]" : ((result.phaseTag == 1) ? "" : " [DBL]");
      proj->SetTitle(Form("p=[%.0f,%.0f]%s", result.pLow, result.pHigh, phaseStr.Data()));
      proj->GetXaxis()->SetRangeUser(fitRangeMin, fitRangeMax);
      proj->GetXaxis()->SetTitle("#Delta#beta");
      proj->GetYaxis()->SetTitle("Counts");
      proj->GetXaxis()->SetTitleSize(0.045);
      proj->GetYaxis()->SetTitleSize(0.045);
      proj->SetLineColor(kBlack);
      proj->SetMarkerStyle(20);
      proj->SetMarkerSize(0.4);
      proj->Draw("E");

      // Zero line
      TLine* zeroLine = new TLine(0, 0, 0, proj->GetMaximum() * 1.1);
      zeroLine->SetLineColor(kGray+1);
      zeroLine->SetLineStyle(kDashed);
      zeroLine->Draw("same");

      if (result.success && result.polyParams.size() > 0) {
        int nPolyParams = result.polyOrder + 1;
        
        // Build total fit expression: gaus(0) + gaus(3) + poly at [6]
        TString funcExpr = "gaus(0) + gaus(3)";
        for (int pp = 0; pp <= result.polyOrder; ++pp) {
          if (pp == 0) funcExpr += Form(" + [%d]", 6 + pp);
          else funcExpr += Form(" + [%d]*pow(x,%d)", 6 + pp, pp);
        }
        
        TF1* fitFunc = new TF1(Form("fitdisp_db_w%d_d%d", w, (int)d), funcExpr, fitRangeMin, fitRangeMax);
        fitFunc->SetParameter(0, result.amplitude);
        fitFunc->SetParameter(1, result.mean);
        fitFunc->SetParameter(2, result.sigma);
        fitFunc->SetParameter(3, result.bkgGausAmp);
        fitFunc->SetParameter(4, result.bkgGausMean);
        fitFunc->SetParameter(5, result.bkgGausSigma);
        for (int pp = 0; pp < nPolyParams; ++pp)
          fitFunc->SetParameter(6 + pp, result.polyParams[pp]);
        fitFunc->SetLineColor(kRed);
        fitFunc->SetLineWidth(2);
        fitFunc->Draw("same");

        // Signal Gaussian (blue solid, thick)
        TF1* sig = new TF1(Form("sig_db_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
        sig->SetParameters(result.amplitude, result.mean, result.sigma);
        sig->SetLineColor(kBlue);
        sig->SetLineStyle(1);
        sig->SetLineWidth(3);
        sig->Draw("same");

        // Background Gaussian (orange dashed) - only draw if amplitude > 0
        if (result.bkgGausAmp > 0.001 * result.amplitude) {
          TF1* bkgGaus = new TF1(Form("bkg_db_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
          bkgGaus->SetParameters(result.bkgGausAmp, result.bkgGausMean, result.bkgGausSigma);
          bkgGaus->SetLineColor(kOrange+1);
          bkgGaus->SetLineStyle(kDashed);
          bkgGaus->SetLineWidth(2);
          bkgGaus->Draw("same");
        }

        // Polynomial (green dashed) - build proper expression
        TString polyExpr;
        for (int pp = 0; pp <= result.polyOrder; ++pp) {
          if (pp == 0) polyExpr = "[0]";
          else polyExpr += Form(" + [%d]*pow(x,%d)", pp, pp);
        }
        TF1* poly = new TF1(Form("poly_db_w%d_d%d", w, (int)d), polyExpr, fitRangeMin, fitRangeMax);
        for (int pp = 0; pp < nPolyParams; ++pp)
          poly->SetParameter(pp, result.polyParams[pp]);
        poly->SetLineColor(kGreen+2);
        poly->SetLineStyle(kDashed);
        poly->SetLineWidth(2);
        poly->Draw("same");
      }

      TLatex tex;
      tex.SetNDC();
      tex.SetTextSize(0.034);
      if (result.phaseTag == 0) {
        tex.SetTextColor(kCyan+2);
        tex.DrawLatex(0.55, 0.87, "BACKWARD");
      } else if (result.phaseTag == 2) {
        tex.SetTextColor(kMagenta+1);
        tex.DrawLatex(0.55, 0.87, "DOUBLING");
      }
      tex.SetTextColor(kBlack);
      
      if (result.success) {
        tex.SetTextColor(kBlue);
        tex.DrawLatex(0.55, 0.79, Form("#mu=%.4f", result.mean));
        tex.DrawLatex(0.55, 0.72, Form("#sigma=%.4f", result.sigma));
        tex.SetTextColor(kOrange+1);
        tex.DrawLatex(0.55, 0.65, Form("bkg=%.0f%%", 100.0 * result.bkgGausAmp / result.amplitude));
        tex.SetTextColor(kBlack);
        tex.DrawLatex(0.55, 0.58, Form("pol%d", result.polyOrder));
        if (result.chi2ndf >= 0.3 && result.chi2ndf <= 4.0) {
          tex.SetTextColor(kGreen+2);
        } else {
          tex.SetTextColor(kOrange+1);
        }
        tex.DrawLatex(0.55, 0.51, Form("#chi^{2}/n=%.1f", result.chi2ndf));
        tex.SetTextColor(kBlack);
      } else {
        tex.SetTextColor(kRed);
        tex.DrawLatex(0.55, 0.77, "Fit failed");
        tex.SetTextColor(kBlack);
      }
      tex.DrawLatex(0.55, 0.44, Form("N=%.0f", result.entries));
      
      gPad->Modified();
      gPad->Update();
    }
    cFits[w]->Modified();
    cFits[w]->Update();
  }

  // =====================================================
  // CONTOUR PLOTS IN Δβ SPACE
  // =====================================================
  
  TCanvas* cCombDB_1sig = new TCanvas("c_comb_db_1sig", "Combined 1#sigma in #Delta#beta", 1000, 800);
  cCombDB_1sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB1 = new TLine(0, 0, 1400, 0);
  zeroLineDB1->SetLineColor(kBlack);
  zeroLineDB1->SetLineStyle(kDashed);
  zeroLineDB1->SetLineWidth(2);
  zeroLineDB1->Draw("same");
  
  TLegend* legDB_1 = new TLegend(0.60, 0.15, 0.88, 0.40);
  legDB_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPoints, dbPoints;
    for (size_t i = 0; i < rawP.size(); ++i) {
      pPoints.push_back(rawP[i]);
      dbPoints.push_back(smoothMean[i] + 1.0 * smoothSigma[i]);
    }
    for (int i = (int)rawP.size() - 1; i >= 0; --i) {
      pPoints.push_back(rawP[i]);
      dbPoints.push_back(smoothMean[i] - 1.0 * smoothSigma[i]);
    }
    
    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), dbPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(4);
      g->Draw("L");
      legDB_1->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legDB_1->AddEntry(zeroLineDB1, "#Delta#beta=0 (proton)", "l");
  legDB_1->Draw();
  cCombDB_1sig->Modified();
  cCombDB_1sig->Update();

  // 3σ in Δβ
  TCanvas* cCombDB_3sig = new TCanvas("c_comb_db_3sig", "Combined 3#sigma in #Delta#beta", 1000, 800);
  cCombDB_3sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB3 = new TLine(0, 0, 1400, 0);
  zeroLineDB3->SetLineColor(kBlack);
  zeroLineDB3->SetLineStyle(kDashed);
  zeroLineDB3->SetLineWidth(2);
  zeroLineDB3->Draw("same");
  
  TLegend* legDB_3 = new TLegend(0.60, 0.15, 0.88, 0.40);
  legDB_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPoints, dbPoints;
    for (size_t i = 0; i < rawP.size(); ++i) {
      pPoints.push_back(rawP[i]);
      dbPoints.push_back(smoothMean[i] + 3.0 * smoothSigma[i]);
    }
    for (int i = (int)rawP.size() - 1; i >= 0; --i) {
      pPoints.push_back(rawP[i]);
      dbPoints.push_back(smoothMean[i] - 3.0 * smoothSigma[i]);
    }
    
    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), dbPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(4);
      g->Draw("L");
      legDB_3->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legDB_3->AddEntry(zeroLineDB3, "#Delta#beta=0 (pion)", "l");
  legDB_3->Draw();
  cCombDB_3sig->Modified();
  cCombDB_3sig->Update();

  // =====================================================
  // TRANSFORM TO (p, β) SPACE
  // =====================================================
  
  TCanvas* cCombPB_1sig = new TCanvas("c_comb_pb_1sig", "Combined 1#sigma in (p, #beta)", 1000, 800);
  cCombPB_1sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB = new TF1("pionCurvePB", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB->SetLineColor(kBlack);
  pionCurvePB->SetLineStyle(kDashed);
  pionCurvePB->SetLineWidth(2);
  pionCurvePB->Draw("same");
  
  TLegend* legPB_1 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legPB_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, betaPointsUpper;
    std::vector<double> pPointsLower, betaPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      if (beta_upper > 0.3 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      if (beta_lower > 0.3 && beta_lower < 1.15) {
        pPointsLower.push_back(p);
        betaPointsLower.push_back(beta_lower);
      }
    }

    if (!pPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(pPointsUpper.size(), pPointsUpper.data(), betaPointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legPB_1->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPointsLower.empty()) {
      TGraph* gLower = new TGraph(pPointsLower.size(), pPointsLower.data(), betaPointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legPB_1->AddEntry(pionCurvePB, "m_{#pi}=139.57", "l");
  legPB_1->Draw();
  cCombPB_1sig->Modified();
  cCombPB_1sig->Update();

  // 3σ in (p, β)
  TCanvas* cCombPB_3sig = new TCanvas("c_comb_pb_3sig", "Combined 3#sigma in (p, #beta)", 1000, 800);
  cCombPB_3sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB3 = new TF1("pionCurvePB3", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB3->SetLineColor(kBlack);
  pionCurvePB3->SetLineStyle(kDashed);
  pionCurvePB3->SetLineWidth(2);
  pionCurvePB3->Draw("same");
  
  TLegend* legPB_3 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legPB_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, betaPointsUpper;
    std::vector<double> pPointsLower, betaPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 3.0 * smoothSigma[i];
      if (beta_upper > 0.3 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      if (beta_lower > 0.3 && beta_lower < 1.15) {
        pPointsLower.push_back(p);
        betaPointsLower.push_back(beta_lower);
      }
    }

    if (!pPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(pPointsUpper.size(), pPointsUpper.data(), betaPointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legPB_3->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPointsLower.empty()) {
      TGraph* gLower = new TGraph(pPointsLower.size(), pPointsLower.data(), betaPointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legPB_3->AddEntry(pionCurvePB3, "m_{#pi}=139.57", "l");
  legPB_3->Draw();
  cCombPB_3sig->Modified();
  cCombPB_3sig->Update();

  // =====================================================
  // TRANSFORM TO (p, mass²) SPACE
  // =====================================================
  
  // Helper lambda: compute mass² from p and beta
  auto mass2FromPBeta = [](double p, double beta) -> double {
    if (beta <= 0.01 || beta >= 1.5) return -999999;
    return p * p * (1.0 / (beta * beta) - 1.0);
  };
  
  const double pionMass2 = gPionMass * gPionMass;  // ~19479.6 MeV²/c⁴
  
  // 1σ in (p, mass²)
  TCanvas* cCombM2_1sig = new TCanvas("c_comb_m2_1sig", "Combined 1#sigma in (p, mass^{2})", 1000, 800);
  cCombM2_1sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  // Draw pion mass² line
  TLine* pionLineM2_1 = new TLine(0, pionMass2, 1400, pionMass2);
  pionLineM2_1->SetLineColor(kBlack);
  pionLineM2_1->SetLineStyle(kDashed);
  pionLineM2_1->SetLineWidth(2);
  pionLineM2_1->Draw("same");
  
  TLegend* legM2_1 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legM2_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, m2PointsUpper;
    std::vector<double> pPointsLower, m2PointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > -20000 && m2_upper < 60000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > -20000 && m2_lower < 60000) {
        pPointsLower.push_back(p);
        m2PointsLower.push_back(m2_lower);
      }
    }

    if (!pPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(pPointsUpper.size(), pPointsUpper.data(), m2PointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legM2_1->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPointsLower.empty()) {
      TGraph* gLower = new TGraph(pPointsLower.size(), pPointsLower.data(), m2PointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legM2_1->AddEntry(pionLineM2_1, Form("m_{#pi}^{2}=%.0f", pionMass2), "l");
  legM2_1->Draw();
  cCombM2_1sig->Modified();
  cCombM2_1sig->Update();

  // 3σ in (p, mass²)
  TCanvas* cCombM2_3sig = new TCanvas("c_comb_m2_3sig", "Combined 3#sigma in (p, mass^{2})", 1000, 800);
  cCombM2_3sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  // Draw pion mass² line
  TLine* pionLineM2_3 = new TLine(0, pionMass2, 1400, pionMass2);
  pionLineM2_3->SetLineColor(kBlack);
  pionLineM2_3->SetLineStyle(kDashed);
  pionLineM2_3->SetLineWidth(2);
  pionLineM2_3->Draw("same");
  
  TLegend* legM2_3 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legM2_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, m2PointsUpper;
    std::vector<double> pPointsLower, m2PointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 3.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > -20000 && m2_upper < 60000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > -20000 && m2_lower < 60000) {
        pPointsLower.push_back(p);
        m2PointsLower.push_back(m2_lower);
      }
    }

    if (!pPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(pPointsUpper.size(), pPointsUpper.data(), m2PointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legM2_3->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPointsLower.empty()) {
      TGraph* gLower = new TGraph(pPointsLower.size(), pPointsLower.data(), m2PointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legM2_3->AddEntry(pionLineM2_3, Form("m_{#pi}^{2}=%.0f", pionMass2), "l");
  legM2_3->Draw();
  cCombM2_3sig->Modified();
  cCombM2_3sig->Update();

  // =====================================================
  // TRANSFORM TO (mass, a) SPACE
  // =====================================================
  
  TCanvas* cCombMA_1sig = new TCanvas("c_comb_ma_1sig", "Combined 1#sigma in (mass, a)", 1000, 800);
  cCombMA_1sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* pionLineMA = new TLine(gPionMass, 0, gPionMass, 16);
  pionLineMA->SetLineColor(kBlack);
  pionLineMA->SetLineStyle(kDashed);
  pionLineMA->SetLineWidth(2);
  pionLineMA->Draw("same");
  
  TLegend* legMA_1 = new TLegend(0.60, 0.55, 0.88, 0.88);
  legMA_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> massPointsUpper, aPointsUpper;
    std::vector<double> massPointsLower, aPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > 0 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > 0 && mass_l < 300 && a_l > 0 && a_l < 16) {
          massPointsLower.push_back(mass_l);
          aPointsLower.push_back(a_l);
        }
      }
    }

    if (!massPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(massPointsUpper.size(), massPointsUpper.data(), aPointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legMA_1->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!massPointsLower.empty()) {
      TGraph* gLower = new TGraph(massPointsLower.size(), massPointsLower.data(), aPointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legMA_1->AddEntry(pionLineMA, "m_{#pi}=139.57", "l");
  legMA_1->Draw();
  cCombMA_1sig->Modified();
  cCombMA_1sig->Update();

  // 3σ in (mass, a)
  TCanvas* cCombMA_3sig = new TCanvas("c_comb_ma_3sig", "Combined 3#sigma in (mass, a)", 1000, 800);
  cCombMA_3sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* pionLineMA3 = new TLine(gPionMass, 0, gPionMass, 16);
  pionLineMA3->SetLineColor(kBlack);
  pionLineMA3->SetLineStyle(kDashed);
  pionLineMA3->SetLineWidth(2);
  pionLineMA3->Draw("same");
  
  TLegend* legMA_3 = new TLegend(0.60, 0.55, 0.88, 0.88);
  legMA_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;
    
    // CRITICAL: Sort by momentum before smoothing!
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> massPointsUpper, aPointsUpper;
    std::vector<double> massPointsLower, aPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 3.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > 0 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > 0 && mass_l < 300 && a_l > 0 && a_l < 16) {
          massPointsLower.push_back(mass_l);
          aPointsLower.push_back(a_l);
        }
      }
    }

    if (!massPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(massPointsUpper.size(), massPointsUpper.data(), aPointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(4);
      gUpper->Draw("L");
      legMA_3->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!massPointsLower.empty()) {
      TGraph* gLower = new TGraph(massPointsLower.size(), massPointsLower.data(), aPointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(4);
      gLower->Draw("L");
    }
  }
  legMA_3->AddEntry(pionLineMA3, "m_{#pi}=139.57", "l");
  legMA_3->Draw();
  cCombMA_3sig->Modified();
  cCombMA_3sig->Update();

  // =====================================================
  // PARAMETER PLOTS
  // =====================================================
  
  TCanvas* cPar = new TCanvas("c_par", "Fit Parameters vs Momentum", 1200, 800);
  cPar->Divide(2, 2);

  // Mean
  cPar->cd(1);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgMean = new TMultiGraph();
  TLegend* legMean = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].mean);
      }
    }
    g->SetMarkerStyle(20);
    g->SetMarkerSize(0.3);
    g->SetMarkerColor(widthColors[w]);
    g->SetLineColor(widthColors[w]);
    mgMean->Add(g, "P");
    legMean->AddEntry(g, widthLabels[w], "p");
  }
  mgMean->SetTitle("Mean #Delta#beta vs Momentum;p [MeV/c];#mu (#Delta#beta)");
  mgMean->Draw("A");
  mgMean->GetYaxis()->SetRangeUser(-0.02, 0.02);
  TLine* zeroMean = new TLine(0, 0, 1400, 0);
  zeroMean->SetLineColor(kRed);
  zeroMean->SetLineStyle(kDashed);
  zeroMean->Draw("same");
  legMean->Draw();

  // Sigma
  cPar->cd(2);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgSigma = new TMultiGraph();
  TLegend* legSigma = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].sigma);
      }
    }
    g->SetMarkerStyle(20);
    g->SetMarkerSize(0.3);
    g->SetMarkerColor(widthColors[w]);
    g->SetLineColor(widthColors[w]);
    mgSigma->Add(g, "P");
    legSigma->AddEntry(g, widthLabels[w], "p");
  }
  mgSigma->SetTitle("Width #sigma vs Momentum;p [MeV/c];#sigma (#Delta#beta)");
  mgSigma->Draw("A");
  legSigma->Draw();

  // Background Gauss fraction
  cPar->cd(3);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgBkg = new TMultiGraph();
  TLegend* legBkg = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success && allFitResults[w][i].amplitude > 0) {
        double frac = 100.0 * allFitResults[w][i].bkgGausAmp / allFitResults[w][i].amplitude;
        g->SetPoint(np++, allMomCenters[w][i], frac);
      }
    }
    g->SetMarkerStyle(20);
    g->SetMarkerSize(0.3);
    g->SetMarkerColor(widthColors[w]);
    g->SetLineColor(widthColors[w]);
    mgBkg->Add(g, "P");
    legBkg->AddEntry(g, widthLabels[w], "p");
  }
  mgBkg->SetTitle("Background Gauss Fraction;p [MeV/c];Bkg/Signal [%]");
  mgBkg->Draw("A");
  mgBkg->GetYaxis()->SetRangeUser(0, 40);
  legBkg->Draw();

  // Chi2/ndf
  cPar->cd(4);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgChi2 = new TMultiGraph();
  TLegend* legChi2 = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].chi2ndf);
      }
    }
    g->SetMarkerStyle(20);
    g->SetMarkerSize(0.3);
    g->SetMarkerColor(widthColors[w]);
    g->SetLineColor(widthColors[w]);
    mgChi2->Add(g, "P");
    legChi2->AddEntry(g, widthLabels[w], "p");
  }
  mgChi2->SetTitle("#chi^{2}/ndf vs Momentum;p [MeV/c];#chi^{2}/ndf");
  mgChi2->Draw("A");
  mgChi2->GetYaxis()->SetRangeUser(0, 5);
  TLine* chi2Line = new TLine(0, 1, 1400, 1);
  chi2Line->SetLineColor(kRed);
  chi2Line->SetLineStyle(kDashed);
  chi2Line->Draw("same");
  legChi2->Draw();

  // =====================================================
  // SAVE OUTPUT
  // =====================================================
  
  for (int w = 0; w < nWidths; ++w) {
    cFits[w]->SaveAs(Form("pion_fits_dBeta_%s.png", widthLabels[w].Data()));
  }
  cCombDB_1sig->SaveAs("pion_dBeta_combined_1sigma.png");
  cCombDB_3sig->SaveAs("pion_dBeta_combined_3sigma.png");
  cCombPB_1sig->SaveAs("pion_pbeta_combined_1sigma.png");
  cCombPB_3sig->SaveAs("pion_pbeta_combined_3sigma.png");
  cCombM2_1sig->SaveAs("pion_mass2_combined_1sigma.png");
  cCombM2_3sig->SaveAs("pion_mass2_combined_3sigma.png");
  cCombMA_1sig->SaveAs("pion_massa_combined_1sigma.png");
  cCombMA_3sig->SaveAs("pion_massa_combined_3sigma.png");
  cPar->SaveAs("pion_parameters_dBeta.png");

  // Output file
  std::ofstream outfile("pion_fit_results_dBeta.txt");
  outfile << "Width\tpCenter\tpLow\tpHigh\tPhase\tMean_dBeta\tSigma_dBeta\tBkgFrac[%]\tChi2NDF\tStatus" << std::endl;
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allMomCenters[w].size(); ++i) {
      FitResult& r = allFitResults[w][i];
      TString phaseStr = (r.phaseTag == 0) ? "BWD" : ((r.phaseTag == 1) ? "FWD" : "DBL");
      double bkgFrac = (r.amplitude > 0) ? 100.0 * r.bkgGausAmp / r.amplitude : 0;
      outfile << widthLabels[w] << "\t" << r.pCenter << "\t" << r.pLow << "\t" << r.pHigh << "\t"
              << phaseStr << "\t" << r.mean << "\t" << r.sigma << "\t" << bkgFrac << "\t" << r.chi2ndf << "\t"
              << (r.success ? "OK" : "FAIL") << std::endl;
    }
  }
  outfile.close();

  cout << "\n=== Analysis Complete ===" << endl;
  cout << "IMPROVED FIT MODEL:" << endl;
  cout << "  Stage 1: Signal Gauss + Polynomial (baseline)" << endl;
  cout << "  Stage 2: Add Background Gauss (<=30% of signal)" << endl;
  cout << "           Polynomial constrained around Stage 1 values" << endl;
  cout << "  Parameter propagation between neighboring slices" << endl;
  cout << "\nOutput files saved." << endl;
}
