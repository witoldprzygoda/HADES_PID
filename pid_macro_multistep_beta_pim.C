// pid_macro_multistep_beta_pion.C
// PION PID analysis using Δβ = β_measured - β_pion(p) representation
// 
// ROBUST FITTING STRATEGY:
// 1. Find peak maximum and 80% boundaries in data
// 2. Preliminary Gaussian fit to peak top - ANCHORS signal position
// 3. Full fit: Signal Gauss + Background Gauss + Polynomial
// 4. SUBTRACT polynomial and background Gauss from data
// 5. Final Gaussian fit to cleaned data - USE THESE for contours
//
// Scanning strategy:
// - Warmup: [200, 400] MeV/c
// - Anchor: 180 MeV/c
// - Phase 0: right edge at 180, width doubling going left to 0
// - Phase 1: forward from 180 to 500 with 1 MeV/c steps
// - Phase 2: above 500, progressive doubling with sliding:
//            double width, then slide "prev_width" steps of 1 MeV/c each
//            This ensures smooth parameter propagation at high momentum

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
const double gSSquared = 1e-4;       // For (mass, a) transformation

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
// ROBUST FITTING FUNCTION
// New approach:
//   1. Find peak maximum and 80% boundaries
//   2. Preliminary Gaussian fit to peak top - anchors signal position
//   3. Full fit with all components (signal + bkg Gauss + polynomial)
//   4. Subtract polynomial and background Gauss from data
//   5. Final Gaussian fit to cleaned data - USE THESE for contours
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
  // STEP 1: Find peak maximum and 80% boundaries
  // Search in signal region near Δβ = 0
  // =====================================================
  int bin_sig_lo = proj->FindFixBin(-0.06);
  int bin_sig_hi = proj->FindFixBin(0.06);
  
  // Find smoothed maximum
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

  // Find 80% boundaries (where yield drops to 80% of max)
  double threshold80 = 0.80 * peakVal;
  
  int bin80_lo = peakBin;
  for (int b = peakBin; b >= bin_sig_lo; --b) {
    if (proj->GetBinContent(b) < threshold80) {
      bin80_lo = b + 1;
      break;
    }
    bin80_lo = b;
  }
  
  int bin80_hi = peakBin;
  for (int b = peakBin; b <= bin_sig_hi; ++b) {
    if (proj->GetBinContent(b) < threshold80) {
      bin80_hi = b - 1;
      break;
    }
    bin80_hi = b;
  }
  
  double x80_lo = proj->GetBinCenter(bin80_lo);
  double x80_hi = proj->GetBinCenter(bin80_hi);
  
  // Ensure we have at least 3 bins for preliminary fit
  if (bin80_hi - bin80_lo < 2) {
    bin80_lo = std::max(bin_sig_lo, peakBin - 2);
    bin80_hi = std::min(bin_sig_hi, peakBin + 2);
    x80_lo = proj->GetBinCenter(bin80_lo);
    x80_hi = proj->GetBinCenter(bin80_hi);
  }

  // =====================================================
  // STEP 2: Preliminary Gaussian fit to peak top only
  // This ANCHORS where the signal must be
  // =====================================================
  TF1* prelimFit = new TF1(Form("prelim_w%d_s%d", widthIdx, sliceIdx), "gaus", x80_lo, x80_hi);
  prelimFit->SetParameter(0, peakVal);
  prelimFit->SetParameter(1, peakPos);
  prelimFit->SetParameter(2, 0.01);
  prelimFit->SetParLimits(1, x80_lo, x80_hi);  // Mean MUST be within 80% region
  prelimFit->SetParLimits(2, 0.002, 0.05);
  
  TFitResultPtr prelimRes = proj->Fit(prelimFit, "SQR0B");
  
  double anchorMean = peakPos;
  double anchorSigma = 0.015;
  if (prelimRes.Get() && prelimRes->IsValid()) {
    anchorMean = prelimFit->GetParameter(1);
    anchorSigma = prelimFit->GetParameter(2);
  }
  delete prelimFit;

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

  // Use propagated params for initialization, but anchor from preliminary fit
  double initSigMean = anchorMean;  // From preliminary fit!
  double initSigSigma = anchorSigma;
  double initSigAmp = sigAmpEst;
  
  if (prevParams.valid) {
    // Use propagated sigma as guide, but mean from preliminary fit
    initSigSigma = prevParams.sigSigma;
    double ampScale = integral / 1000.0;
    initSigAmp = prevParams.sigAmp * ampScale;
    if (initSigAmp < 0.5) initSigAmp = sigAmpEst;
    if (initSigAmp > 10.0 * peakVal) initSigAmp = sigAmpEst;
  }

  // =====================================================
  // STEP 3: Full fit with all components
  // Try polynomial orders 2, 3, 4
  // =====================================================
  struct FullFitResult {
    bool valid;
    double chi2ndf;
    double sigMean, sigSigma, sigAmp;
    double bkgAmp, bkgMean, bkgSigma;
    std::vector<double> polyParams;
    int polyOrder;
  };
  
  const int polyOrders[3] = {2, 3, 4};
  const int nModels = 3;
  FullFitResult results[nModels];
  for (int i = 0; i < nModels; ++i) results[i].valid = false;

  for (int m = 0; m < nModels; ++m) {
    int polyOrder = polyOrders[m];
    int nPolyParams = polyOrder + 1;
    
    // Build expression: gaus(0) + gaus(3) + poly starting at [6]
    TString funcExpr = "gaus(0) + gaus(3)";
    for (int pp = 0; pp <= polyOrder; ++pp) {
      if (pp == 0) funcExpr += Form(" + [%d]", 6 + pp);
      else funcExpr += Form(" + [%d]*pow(x,%d)", 6 + pp, pp);
    }
    
    TString funcName = Form("fullfit_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fullFit = new TF1(funcName, funcExpr, fitMin, fitMax);
    
    // Signal Gaussian - CONSTRAINED around anchor position!
    fullFit->SetParameter(0, initSigAmp);
    fullFit->SetParameter(1, anchorMean);  // From preliminary fit
    fullFit->SetParameter(2, initSigSigma);
    fullFit->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal));
    // Mean constrained tightly around anchor
    double meanTol = 0.01;  // ±0.01 around anchor
    fullFit->SetParLimits(1, anchorMean - meanTol, anchorMean + meanTol);
    fullFit->SetParLimits(2, 0.003, 0.06);
    
    // Background Gaussian - constrained to not overshoot data
    // Estimate max reasonable amplitude from data edges
    double edgeMax = std::max(bkgLeft, bkgRight);
    double maxBkgAmp = std::min(0.3 * initSigAmp, 2.0 * edgeMax);  // Can't exceed 2x edge level
    if (maxBkgAmp < 0.01 * initSigAmp) maxBkgAmp = 0.1 * initSigAmp;  // But allow some minimum
    
    fullFit->SetParameter(3, 0.05 * initSigAmp);
    fullFit->SetParLimits(3, 0.0, maxBkgAmp);
    fullFit->SetParameter(4, 0.06);  // Mean offset from signal
    fullFit->SetParLimits(4, 0.03, 0.12);
    fullFit->SetParameter(5, 0.05);  // Sigma - moderate width
    fullFit->SetParLimits(5, 0.025, 0.10);
    
    // Polynomial
    if (prevParams.valid && prevParams.polyParams.size() > 0) {
      for (int pp = 0; pp < nPolyParams && pp < (int)prevParams.polyParams.size(); ++pp)
        fullFit->SetParameter(6 + pp, prevParams.polyParams[pp]);
      for (int pp = prevParams.polyParams.size(); pp < nPolyParams; ++pp)
        fullFit->SetParameter(6 + pp, 0.0);
    } else {
      fullFit->SetParameter(6, std::max(0.1, bkgAvg));
      double slope = (bkgRight - bkgLeft) / (fitMax - fitMin);
      if (polyOrder >= 1) fullFit->SetParameter(7, slope);
      for (int pp = 2; pp < nPolyParams; ++pp)
        fullFit->SetParameter(6 + pp, 0.0);
    }

    TFitResultPtr res = proj->Fit(fullFit, "SQR0B");
    
    if (!res.Get() || !res->IsValid() || res->Status() != 0) {
      delete fullFit;
      continue;
    }
    
    double chi2ndf = (res->Ndf() > 0) ? res->Chi2() / res->Ndf() : 1e6;
    
    // Extract fit results
    double f_sigAmp = fullFit->GetParameter(0);
    double f_sigMean = fullFit->GetParameter(1);
    double f_sigSigma = fullFit->GetParameter(2);
    double f_bkgAmp = fullFit->GetParameter(3);
    double f_bkgMean = fullFit->GetParameter(4);
    double f_bkgSigma = fullFit->GetParameter(5);
    std::vector<double> f_poly;
    for (int pp = 0; pp < nPolyParams; ++pp)
      f_poly.push_back(fullFit->GetParameter(6 + pp));
    
    delete fullFit;
    
    // Basic validation
    if (chi2ndf > 20.0) continue;
    if (f_sigSigma < 0.003 || f_sigSigma > 0.07) continue;
    if (f_sigMean < -0.05 || f_sigMean > 0.05) continue;
    if (f_sigAmp < 0.3) continue;
    
    results[m].valid = true;
    results[m].chi2ndf = chi2ndf;
    results[m].sigMean = f_sigMean;
    results[m].sigSigma = f_sigSigma;
    results[m].sigAmp = f_sigAmp;
    results[m].bkgAmp = f_bkgAmp;
    results[m].bkgMean = f_bkgMean;
    results[m].bkgSigma = f_bkgSigma;
    results[m].polyOrder = polyOrder;
    results[m].polyParams = f_poly;
  }

  // Select best model - prefer simpler polynomial if chi2 OK
  const double chi2_good_min = 0.3;
  const double chi2_good_max = 3.0;
  
  int bestIdx = -1;
  for (int m = 0; m < nModels; ++m) {
    if (results[m].valid) {
      double c = results[m].chi2ndf;
      if (c >= chi2_good_min && c <= chi2_good_max) {
        bestIdx = m;
        break;
      }
    }
  }
  if (bestIdx < 0) {
    double bestScore = 1e9;
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
  
  if (bestIdx < 0) return best;  // No valid fit

  // =====================================================
  // STEP 4 & 5: Subtract background and refit signal
  // Create histogram with polynomial and bkg Gauss subtracted
  // Then fit pure Gaussian to get final signal parameters
  // =====================================================
  
  FullFitResult& chosen = results[bestIdx];
  
  // Create subtracted histogram
  TH1D* hSubtracted = (TH1D*)proj->Clone(Form("hsub_w%d_s%d", widthIdx, sliceIdx));
  
  // Build polynomial function
  TF1* polyFunc = nullptr;
  {
    TString polyExpr;
    for (int pp = 0; pp <= chosen.polyOrder; ++pp) {
      if (pp == 0) polyExpr = "[0]";
      else polyExpr += Form(" + [%d]*pow(x,%d)", pp, pp);
    }
    polyFunc = new TF1("polyFunc", polyExpr, fitMin, fitMax);
    for (int pp = 0; pp <= chosen.polyOrder; ++pp)
      polyFunc->SetParameter(pp, chosen.polyParams[pp]);
  }
  
  // Build background Gaussian
  TF1* bkgGausFunc = new TF1("bkgGausFunc", "gaus", fitMin, fitMax);
  bkgGausFunc->SetParameters(chosen.bkgAmp, chosen.bkgMean, chosen.bkgSigma);
  
  // Subtract polynomial and background Gauss from each bin
  for (int b = 1; b <= hSubtracted->GetNbinsX(); ++b) {
    double x = hSubtracted->GetBinCenter(b);
    double origVal = hSubtracted->GetBinContent(b);
    double polyVal = polyFunc->Eval(x);
    double bkgGausVal = bkgGausFunc->Eval(x);
    double newVal = origVal - polyVal - bkgGausVal;
    hSubtracted->SetBinContent(b, newVal);
    // Keep original error
  }
  
  delete polyFunc;
  delete bkgGausFunc;
  
  // =====================================================
  // STEP 5: Final Gaussian fit to cleaned data
  // These are the parameters we use for contours!
  // =====================================================
  
  // Fit range around the signal peak
  double finalFitMin = anchorMean - 3.0 * chosen.sigSigma;
  double finalFitMax = anchorMean + 3.0 * chosen.sigSigma;
  if (finalFitMin < fitMin) finalFitMin = fitMin;
  if (finalFitMax > fitMax) finalFitMax = fitMax;
  
  TF1* finalGaus = new TF1(Form("finalGaus_w%d_s%d", widthIdx, sliceIdx), "gaus", finalFitMin, finalFitMax);
  finalGaus->SetParameter(0, chosen.sigAmp);
  finalGaus->SetParameter(1, chosen.sigMean);
  finalGaus->SetParameter(2, chosen.sigSigma);
  finalGaus->SetParLimits(1, anchorMean - 0.02, anchorMean + 0.02);
  finalGaus->SetParLimits(2, 0.003, 0.06);
  
  TFitResultPtr finalRes = hSubtracted->Fit(finalGaus, "SQR0B");
  
  double finalMean = chosen.sigMean;
  double finalSigma = chosen.sigSigma;
  double finalAmp = chosen.sigAmp;
  double finalChi2ndf = chosen.chi2ndf;
  
  if (finalRes.Get() && finalRes->IsValid() && finalRes->Status() == 0) {
    finalMean = finalGaus->GetParameter(1);
    finalSigma = finalGaus->GetParameter(2);
    finalAmp = finalGaus->GetParameter(0);
    if (finalRes->Ndf() > 0) {
      finalChi2ndf = finalRes->Chi2() / finalRes->Ndf();
    }
    
    // Sanity check - if final fit gives crazy values, use full fit values
    if (finalSigma < 0.003 || finalSigma > 0.07 || 
        std::abs(finalMean - anchorMean) > 0.03) {
      finalMean = chosen.sigMean;
      finalSigma = chosen.sigSigma;
      finalAmp = chosen.sigAmp;
      finalChi2ndf = chosen.chi2ndf;
    }
  }
  
  delete finalGaus;
  delete hSubtracted;
  
  // =====================================================
  // Store results - use FINAL fit parameters for contours
  // =====================================================
  best.success = true;
  best.mean = finalMean;
  best.sigma = finalSigma;
  best.amplitude = finalAmp;
  best.polyOrder = chosen.polyOrder;
  best.chi2ndf = finalChi2ndf;
  best.bkgGausAmp = chosen.bkgAmp;
  best.bkgGausMean = chosen.bkgMean;
  best.bkgGausSigma = chosen.bkgSigma;
  best.polyParams = chosen.polyParams;

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

void pid_macro_multistep_beta_pim() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain
  const char* treeName = "PimEpEm";
  const std::vector<TString> files = {
	  "pp060.root","pp049.root"
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
  
  // Main histogram: p vs Δβ (pion)
  const char* h2name = "h2_p_deltaBeta";
  TString drawCmd = Form(
    "pim_beta - pim_p/sqrt(pim_p*pim_p + %.2f*%.2f) : pim_p >> %s(280,0,1400,300,-0.15,0.15)",
    gPionMass, gPionMass, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "isBest==1", "colz");
  TH2F* h2DB = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2DB) {
    cout << "Failed to create histogram" << endl;
    return;
  }

  h2DB->SetTitle("Momentum vs #Delta#beta (#beta - #beta_{p})");
  h2DB->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DB->GetYaxis()->SetTitle("#Delta#beta = #beta - #beta_{p}");

  // (p, β) histogram
  const char* h2pb_name = "h2_pim_beta";
  if (gDirectory->FindObject(h2pb_name)) gDirectory->Delete(Form("%s;*", h2pb_name));
  chain->Draw(Form("pim_beta : pim_p >> %s(280,0,1400,300,0.3,1.15)", h2pb_name), "isBest==1", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2pb_name));

  // (mass, a) histogram
  const char* h2ma_name = "h2_mass_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) >> %s(300,0,300,160,0,16)",
    gSSquared, h2ma_name);
  if (gDirectory->FindObject(h2ma_name)) gDirectory->Delete(Form("%s;*", h2ma_name));
  chain->Draw(drawMA, "isBest==1 && pim_beta>0 && pim_beta<1", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2ma_name));

  // (p, mass²) histogram - mass² = p² * (1/β² - 1)
  const char* h2m2_name = "h2_p_mass2";
  if (gDirectory->FindObject(h2m2_name)) gDirectory->Delete(Form("%s;*", h2m2_name));
  chain->Draw(Form("pim_p*pim_p*(1.0/(pim_beta*pim_beta) - 1) : pim_p >> %s(280,0,1400,400,-20000,60000)", h2m2_name), 
              "isBest==1 && pim_beta>0.1 && pim_beta<1.5", "colz");
  TH2F* h2M2 = static_cast<TH2F*>(gDirectory->Get(h2m2_name));
  if (h2M2) {
    h2M2->SetTitle("Mass^{2} vs Momentum (Pion)");
    h2M2->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2M2->GetYaxis()->SetTitle("Mass^{2} [MeV^{2}/c^{4}]");
  }

  // =====================================================
  // SCANNING PARAMETERS - PION
  // =====================================================
  
  const double warmupLow = 220.0;       // Warmup fit range [220, 400] - always good
  const double warmupHigh = 280.0;
  const double startMom = 120.0;        // Anchor point for regular fitting
  const double transitionMom = 700.0;   // Where Phase 2 doubling starts
  const double endMom = 1500.0;         // Extended range for pions
  const double stepSize = 1.0;          // 1 MeV/c steps in Phase 1
  
  const int nWidths = 4;
  const double baseWidths[nWidths] = {10.0, 20.0, 30.0, 40.0};  // Widths for pions
  const TString widthLabels[nWidths] = {"1x(10)", "2x(20)", "3x(30)", "4x(40)"};
  const int widthColors[nWidths] = {kBlue, kRed, kGreen+2, kMagenta};

  const double fitRangeMin = -0.12;
  const double fitRangeMax = 0.12;

  std::vector<std::vector<FitResult>> allFitResults(nWidths);
  std::vector<std::vector<double>> allMomCenters(nWidths);

  TCanvas* cFits[4];
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
    // Phase 0: RIGHT EDGE anchored at startMom (180), width DOUBLES going left
    // Phase 1: forward from startMom to transitionMom with 1 MeV/c steps
    // Phase 2: forward from transitionMom with doubling width
    // =====================================================
    
    std::vector<double> momLows, momHighs, momCenters, sliceWidths;
    std::vector<int> phaseTag;

    // Phase 0: RIGHT EDGE anchored at startMom (180), width DOUBLES going left
    // For baseW=5: [175,180], [170,180], [160,180], [140,180], [100,180], [0,180]
    // Width: 10 → 20 → 40 → 80 → ... until left reaches 0
    std::vector<int> phase0Indices;
    {
      double pRight = startMom;  // Always 180
      double currentWidth = baseW;
      double pLeft = pRight - currentWidth;
      
      while (pLeft > 0) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);  // Always startMom (180)
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(0);  // Phase 0
        phase0Indices.push_back(idx);
        
        // Double the width for next slice
        currentWidth *= 2.0;
        pLeft = pRight - currentWidth;
      }
      
      // Final slice from 0 to startMom
      int idx = momLows.size();
      momLows.push_back(0);
      momHighs.push_back(pRight);  // 180
      momCenters.push_back(0.5 * pRight);
      sliceWidths.push_back(pRight);
      phaseTag.push_back(0);
      phase0Indices.push_back(idx);
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

    // Phase 2: Progressive doubling with sliding windows
    // - Double width, then slide "previous width" number of 1 MeV/c steps
    // - This ensures smooth parameter propagation
    // For baseW=5: [500,510]→slide 5→[505,515], double→[505,525]→slide 10→[515,535], etc.
    std::vector<int> phase2Indices;
    {
      double pLeft = transitionMom;  // 500
      double prevWidth = baseW;      // Width at end of Phase 1
      double currentWidth = baseW * 2.0;  // Start with doubled width
      
      while (pLeft + currentWidth <= endMom) {
        // Number of 1 MeV/c steps = previous width value
        int nSteps = static_cast<int>(prevWidth);
        
        for (int step = 0; step <= nSteps && pLeft + currentWidth <= endMom; ++step) {
          int idx = momLows.size();
          momLows.push_back(pLeft);
          momHighs.push_back(pLeft + currentWidth);
          momCenters.push_back(pLeft + 0.5 * currentWidth);
          sliceWidths.push_back(currentWidth);
          phaseTag.push_back(2);
          phase2Indices.push_back(idx);
          
          if (step < nSteps) pLeft += stepSize;  // 1 MeV/c step
        }
        
        // After sliding, double the width for next round
        prevWidth = currentWidth;
        currentWidth *= 2.0;
      }
      
      // Final slice to reach endMom if needed
      if (pLeft < endMom && pLeft + prevWidth < endMom) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(endMom);
        momCenters.push_back(0.5 * (pLeft + endMom));
        sliceWidths.push_back(endMom - pLeft);
        phaseTag.push_back(2);
        phase2Indices.push_back(idx);
      }
    }
    int nPhase2 = phase2Indices.size();

    int nSlices = momLows.size();
    
    // Print Phase 0 slice structure
    cout << "Phase 0 slices (right edge at " << startMom << ", width doubling):" << endl;
    for (size_t k = 0; k < phase0Indices.size(); ++k) {
      int idx = phase0Indices[k];
      cout << Form("  [%3.0f, %3.0f] width=%3.0f", 
                   momLows[idx], momHighs[idx], sliceWidths[idx]) << endl;
    }
    
    cout << "Slices: Phase0=" << nPhase0 << " (width doubling below " << startMom << "), Phase1=" << nPhase1 
         << " (forward, 1 MeV/c steps), Phase2=" << nPhase2 << " (progressive doubling+sliding), Total=" << nSlices << endl;
    
    // Print Phase 2 structure (show transitions)
    if (nPhase2 > 0) {
      cout << "Phase 2 structure:" << endl;
      double lastWidth = 0;
      for (size_t k = 0; k < phase2Indices.size(); ++k) {
        int idx = phase2Indices[k];
        if (sliceWidths[idx] != lastWidth) {
          cout << Form("  Width %.0f: starting at [%.0f, %.0f]", 
                       sliceWidths[idx], momLows[idx], momHighs[idx]) << endl;
          lastWidth = sliceWidths[idx];
        }
      }
    }

    allMomCenters[w] = momCenters;
    allFitResults[w].resize(nSlices);

    // =====================================================
    // FIT IN CORRECT ORDER:
    // 0. WARMUP: [200, 400] - fixed range, always good, get initial params
    // 1. ANCHOR: First slice of Phase 1 [180, 180+baseW] - use warmup params
    // 2. Phase 0: width doubling below 180 (right edge at 180), uses warmup params
    // 3. Rest of Phase 1 forward from anchor to 500
    // 4. Phase 2 forward with doubling above 500
    // =====================================================
    
    int nSuccess = 0;
    PropagatedParams warmupParams;
    warmupParams.valid = false;
    PropagatedParams anchorParams;
    anchorParams.valid = false;
    
    // Step 0: WARMUP FIT [200, 400] - this always works well!
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
    
    // Step 2: Fit Phase 0 using WARMUP parameters (width doubling below 180)
    cout << "Fitting Phase 0 (width doubling below " << startMom << ", using warmup params)..." << endl;
    PropagatedParams prevParams = warmupParams;  // USE WARMUP PARAMS DIRECTLY!
    
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
        
        cout << Form("  P0[%2zu] [%3.0f,%3.0f] w=%3.0f: μ=%7.4f, σ=%6.4f, χ²=%.2f",
                     k, momLows[i], momHighs[i], sliceWidths[i],
                     allFitResults[w][i].mean, allFitResults[w][i].sigma,
                     allFitResults[w][i].chi2ndf) << endl;
      } else {
        cout << Form("  P0[%2zu] [%3.0f,%3.0f] w=%3.0f: FAILED (entries=%.0f)",
                     k, momLows[i], momHighs[i], sliceWidths[i], 
                     allFitResults[w][i].entries) << endl;
      }
      
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
    // Show consistent momentum ranges across all widths:
    // - Phase 0 fits (below 180)
    // - Representative momenta: ~250, ~350, ~450, ~550, ~700, ~900, ~1100, ~1300
    // =====================================================
    std::vector<int> displayIndices;
    
    // From Phase 0: show a few (these are the low-momentum wide slices)
    if (nPhase0 > 0) {
      displayIndices.push_back(phase0Indices[0]);  // First (narrowest)
      if (nPhase0 > 2) displayIndices.push_back(phase0Indices[nPhase0/2]);  // Middle
      displayIndices.push_back(phase0Indices[nPhase0-1]);  // Last (widest, includes 0)
    }
    
    // From Phase 1 & 2: select by target momentum centers
    // These targets ensure we show similar regions for all widths
    std::vector<double> targetMomenta = {250, 350, 450, 550, 700, 900, 1100, 1300};
    
    for (double target : targetMomenta) {
      int bestIdx = -1;
      double bestDist = 1e9;
      
      // Search in Phase 1
      for (size_t k = 0; k < phase1Indices.size(); ++k) {
        int idx = phase1Indices[k];
        double dist = std::abs(momCenters[idx] - target);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      }
      
      // Search in Phase 2
      for (size_t k = 0; k < phase2Indices.size(); ++k) {
        int idx = phase2Indices[k];
        double dist = std::abs(momCenters[idx] - target);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      }
      
      // Accept if within 100 MeV/c of target
      if (bestIdx >= 0 && bestDist < 100) {
        displayIndices.push_back(bestIdx);
      }
    }
    
    // Remove duplicates and limit to 12
    std::sort(displayIndices.begin(), displayIndices.end());
    displayIndices.erase(std::unique(displayIndices.begin(), displayIndices.end()), displayIndices.end());
    if (displayIndices.size() > (size_t)nDisplayPerWidth)
      displayIndices.resize(nDisplayPerWidth);

    cFits[w] = new TCanvas(Form("c_fits_dBeta_%d", w), 
                           Form("Delta-Beta Fits (Pion) - %s", widthLabels[w].Data()), 1400, 900);
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
  
  TCanvas* cCombDB_1sig = new TCanvas("c_comb_db_1sig", "Combined 1#sigma in #Delta#beta (Pion)", 1000, 800);
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
    
    // Mean position line (solid, thin)
    if (!rawP.empty()) {
      TGraph* gMeanLine = new TGraph(rawP.size(), rawP.data(), smoothMean.data());
      gMeanLine->SetLineColor(widthColors[w]);
      gMeanLine->SetLineWidth(2);
      gMeanLine->SetLineStyle(1);  // Solid for mean
      gMeanLine->Draw("L");
    }
    
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
      g->SetLineWidth(3);
      g->SetLineStyle(kDashed);
      g->Draw("L");
      legDB_1->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legDB_1->AddEntry(zeroLineDB1, "#Delta#beta=0 (pion)", "l");
  legDB_1->Draw();
  cCombDB_1sig->Modified();
  cCombDB_1sig->Update();

  // 3σ in Δβ
  TCanvas* cCombDB_3sig = new TCanvas("c_comb_db_3sig", "Combined 3#sigma in #Delta#beta (Pion)", 1000, 800);
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
    
    // Mean position line (solid, thin)
    if (!rawP.empty()) {
      TGraph* gMeanLine = new TGraph(rawP.size(), rawP.data(), smoothMean.data());
      gMeanLine->SetLineColor(widthColors[w]);
      gMeanLine->SetLineWidth(2);
      gMeanLine->SetLineStyle(1);  // Solid for mean
      gMeanLine->Draw("L");
    }
    
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
      g->SetLineWidth(3);
      g->SetLineStyle(kDashed);
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
  
  // 1σ in (p, β)
  TCanvas* cCombPB_1sig = new TCanvas("c_comb_pb_1sig", "Combined 1#sigma in (p, #beta) Pion", 1000, 800);
  cCombPB_1sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB = new TF1("pionCurvePB", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB->SetLineColor(kBlack);
  pionCurvePB->SetLineStyle(kDashed);
  pionCurvePB->SetLineWidth(2);
  pionCurvePB->Draw("same");
  
  TLegend* legPB_1 = new TLegend(0.55, 0.12, 0.88, 0.40);
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
    
    // Mean position line
    std::vector<double> pPointsMean, betaPointsMean;
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      double beta_mean = beta_pion + smoothMean[i];
      if (beta_mean > 0.1 && beta_mean < 1.15) {
        pPointsMean.push_back(p);
        betaPointsMean.push_back(beta_mean);
      }
    }
    
    if (!pPointsMean.empty()) {
      TGraph* gMeanLine = new TGraph(pPointsMean.size(), pPointsMean.data(), betaPointsMean.data());
      gMeanLine->SetLineColor(widthColors[w]);
      gMeanLine->SetLineWidth(2);
      gMeanLine->SetLineStyle(1);  // Solid for mean
      gMeanLine->Draw("L");
    }
    
    // Upper and lower 1σ bounds
    std::vector<double> pPointsUpper, betaPointsUpper;
    std::vector<double> pPointsLower, betaPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      if (beta_lower > 0.1 && beta_lower < 1.15) {
        pPointsLower.push_back(p);
        betaPointsLower.push_back(beta_lower);
      }
    }

    if (!pPointsUpper.empty()) {
      TGraph* gUpper = new TGraph(pPointsUpper.size(), pPointsUpper.data(), betaPointsUpper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(3);
      gUpper->SetLineStyle(kDashed);
      gUpper->Draw("L");
      legPB_1->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPointsLower.empty()) {
      TGraph* gLower = new TGraph(pPointsLower.size(), pPointsLower.data(), betaPointsLower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(3);
      gLower->SetLineStyle(kDashed);
      gLower->Draw("L");
    }
  }
  legPB_1->AddEntry(pionCurvePB, "m_{#pi}=139.57", "l");
  legPB_1->Draw();
  cCombPB_1sig->Modified();
  cCombPB_1sig->Update();

  // 3σ and 5σ in (p, β)
  TCanvas* cCombPB_3sig = new TCanvas("c_comb_pb_3sig", "Combined 3#sigma and 5#sigma in (p, #beta) Pion", 1000, 800);
  cCombPB_3sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB3 = new TF1("pionCurvePB3", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB3->SetLineColor(kBlack);
  pionCurvePB3->SetLineStyle(kDashed);
  pionCurvePB3->SetLineWidth(2);
  pionCurvePB3->Draw("same");
  
  TLegend* legPB_3 = new TLegend(0.55, 0.12, 0.88, 0.45);
  legPB_3->SetHeader("3#sigma (dashed), 5#sigma (long dash)");

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
    
    // Mean position line (solid, thin)
    std::vector<double> pPointsMean, betaPointsMean;
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      double beta_mean = beta_pion + smoothMean[i];
      if (beta_mean > 0.1 && beta_mean < 1.15) {
        pPointsMean.push_back(p);
        betaPointsMean.push_back(beta_mean);
      }
    }
    
    if (!pPointsMean.empty()) {
      TGraph* gMeanLine = new TGraph(pPointsMean.size(), pPointsMean.data(), betaPointsMean.data());
      gMeanLine->SetLineColor(widthColors[w]);
      gMeanLine->SetLineWidth(2);
      gMeanLine->SetLineStyle(1);  // Solid for mean
      gMeanLine->Draw("L");
    }
    
    // 3σ bounds (dashed)
    std::vector<double> pPoints3Upper, betaPoints3Upper;
    std::vector<double> pPoints3Lower, betaPoints3Lower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 3.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints3Upper.push_back(p);
        betaPoints3Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      if (beta_lower > 0.1 && beta_lower < 1.15) {
        pPoints3Lower.push_back(p);
        betaPoints3Lower.push_back(beta_lower);
      }
    }

    if (!pPoints3Upper.empty()) {
      TGraph* gUpper = new TGraph(pPoints3Upper.size(), pPoints3Upper.data(), betaPoints3Upper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(3);
      gUpper->SetLineStyle(kDashed);  // Short dashed for 3σ
      gUpper->Draw("L");
      legPB_3->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPoints3Lower.empty()) {
      TGraph* gLower = new TGraph(pPoints3Lower.size(), pPoints3Lower.data(), betaPoints3Lower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(3);
      gLower->SetLineStyle(kDashed);
      gLower->Draw("L");
    }
    
    // 5σ bounds (long dashed)
    std::vector<double> pPoints5Upper, betaPoints5Upper;
    std::vector<double> pPoints5Lower, betaPoints5Lower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      double beta_upper = beta_pion + smoothMean[i] + 5.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints5Upper.push_back(p);
        betaPoints5Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 5.0 * smoothSigma[i];
      if (beta_lower > 0.1 && beta_lower < 1.15) {
        pPoints5Lower.push_back(p);
        betaPoints5Lower.push_back(beta_lower);
      }
    }

    if (!pPoints5Upper.empty()) {
      TGraph* g5Upper = new TGraph(pPoints5Upper.size(), pPoints5Upper.data(), betaPoints5Upper.data());
      g5Upper->SetLineColor(widthColors[w]);
      g5Upper->SetLineWidth(2);
      g5Upper->SetLineStyle(7);  // Long dashed (style 7) for 5σ
      g5Upper->Draw("L");
    }
    if (!pPoints5Lower.empty()) {
      TGraph* g5Lower = new TGraph(pPoints5Lower.size(), pPoints5Lower.data(), betaPoints5Lower.data());
      g5Lower->SetLineColor(widthColors[w]);
      g5Lower->SetLineWidth(2);
      g5Lower->SetLineStyle(7);  // Long dashed
      g5Lower->Draw("L");
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
  
  const double pionMass2 = gPionMass * gPionMass;  // ~19,480 MeV²/c⁴
  
  // 1σ in (p, mass²)
  TCanvas* cCombM2_1sig = new TCanvas("c_comb_m2_1sig", "Combined 1#sigma in (p, mass^{2}) Pion", 1000, 800);
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
      if (m2_upper > 0 && m2_upper < 2000000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > 0 && m2_lower < 2000000) {
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
  TCanvas* cCombM2_3sig = new TCanvas("c_comb_m2_3sig", "Combined 3#sigma in (p, mass^{2}) Pion", 1000, 800);
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
      if (m2_upper > 0 && m2_upper < 2000000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > 0 && m2_lower < 2000000) {
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
  
  TCanvas* cCombMA_1sig = new TCanvas("c_comb_ma_1sig", "Combined 1#sigma in (mass, a) Pion", 1000, 800);
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
            mass_u > 0 && mass_u < 2000 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 1.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > 0 && mass_l < 2000 && a_l > 0 && a_l < 16) {
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
  TCanvas* cCombMA_3sig = new TCanvas("c_comb_ma_3sig", "Combined 3#sigma in (mass, a) Pion", 1000, 800);
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
            mass_u > 0 && mass_u < 2000 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_pion + smoothMean[i] - 3.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > 0 && mass_l < 2000 && a_l > 0 && a_l < 16) {
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
  
  TCanvas* cPar = new TCanvas("c_par", "Fit Parameters vs Momentum (Pion)", 1200, 800);
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

  cout << "\n=== PION Analysis Complete ===" << endl;
  cout << "ROBUST FIT MODEL:" << endl;
  cout << "  1. Find peak max and 80% boundaries" << endl;
  cout << "  2. Preliminary Gauss fit to peak top (anchors position)" << endl;
  cout << "  3. Full fit: Signal Gauss + Bkg Gauss + Polynomial" << endl;
  cout << "  4. Subtract polynomial and bkg Gauss from data" << endl;
  cout << "  5. Final Gauss fit to cleaned data - USED FOR CONTOURS" << endl;
  cout << "\nScanning strategy:" << endl;
  cout << "  Warmup: [" << warmupLow << ", " << warmupHigh << "] MeV/c" << endl;
  cout << "  Anchor: " << startMom << " MeV/c" << endl;
  cout << "  Phase 0: right edge at " << startMom << ", width doubling to 0" << endl;
  cout << "  Phase 1: forward from " << startMom << " to " << transitionMom << " MeV/c (" << stepSize << " MeV/c steps)" << endl;
  cout << "  Phase 2: above " << transitionMom << " MeV/c - progressive doubling with sliding" << endl;
  cout << "           (double width, then slide prev_width steps of 1 MeV/c each)" << endl;
  cout << "\nOutput files saved." << endl;
}
