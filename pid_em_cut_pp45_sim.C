// pid_em_cut_pp45_sim.C
// ELECTRON (e-) PID analysis using Δβ = β_measured - β_electron(p) representation
// 
// Adapted from pid_pip_cut_pp45_sim.C for HADES experiment
// 
// ELECTRON PHYSICS:
// - Mass: 0.511 MeV/c² (highly relativistic at low momenta)
// - Very narrow Δβ distributions expected due to ultra-relativistic nature
//
// ROBUST FITTING STRATEGY:
// 1. Find peak maximum and 80% boundaries in data
// 2. Preliminary Gaussian fit to peak top - ANCHORS signal position
// 3. Full fit: Signal Gauss + Background Gauss + Polynomial
// 4. SUBTRACT polynomial and background Gauss from data
// 5. Final Gaussian fit to cleaned data - USE THESE for contours
//
// Scanning strategy:
// - Warmup: [130, 160] MeV/c
// - Anchor: 80 MeV/c
// - Phase 0: right edge at anchor, width doubling going left to 0
// - Phase 1: forward from anchor to transition with 1 MeV/c steps
// - Phase 2: above transition, progressive doubling with sliding
// - TCutG extended from p=0 to p=1600 MeV/c via extrapolation

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
#include "TCutG.h"
#include "TSpline.h"
#include "TNamed.h"

using std::cout; using std::endl;

// =====================================================
// GLOBAL CONSTANTS - ELECTRON
// =====================================================
const double gElectronMass = 0.511;   // MeV/c² - electron/positron mass
const double gSSquared = 1e-4;        // For (mass, a) transformation
const double gExtendTo = 1600.0;      // Extend contours to this momentum [MeV/c]
const double gExtendLow = 10.0;       // Extend contours down to this momentum [MeV/c] (avoid distortion at 0)

// =====================================================
// PHYSICS FUNCTIONS - ELECTRON
// =====================================================

double betaElectron(double p) {
  return p / std::sqrt(p * p + gElectronMass * gElectronMass);
}

double deltaBetaToBeta(double p, double dBeta) {
  return betaElectron(p) + dBeta;
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
  double mean;
  double sigma;
  double amplitude;
  int polyOrder;
  double chi2ndf;
  double entries;
  double pCenter, pLow, pHigh, sliceWidth;
  int phaseTag;
  double bkgGausAmp, bkgGausMean, bkgGausSigma;
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

void sortByMomentum(std::vector<double>& p, std::vector<double>& mean, std::vector<double>& sigma) {
  if (p.size() != mean.size() || p.size() != sigma.size()) return;
  
  std::vector<size_t> indices(p.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
  
  std::sort(indices.begin(), indices.end(), 
            [&p](size_t a, size_t b) { return p[a] < p[b]; });
  
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

void extendToMomentum(std::vector<double>& p, std::vector<double>& mean, 
                       std::vector<double>& sigma, double targetP, 
                       double stepP = 20.0, int nAvg = 5) {
  if (p.empty()) return;
  double dataMaxP = p.back();
  if (targetP <= dataMaxP) return;
  
  int actualAvg = std::min(nAvg, (int)p.size());
  double lastMean = 0, lastSigma = 0;
  for (int i = p.size() - actualAvg; i < (int)p.size(); ++i) {
    lastMean += mean[i];
    lastSigma += sigma[i];
  }
  lastMean /= actualAvg;
  lastSigma /= actualAvg;
  
  for (double pExt = dataMaxP + stepP; pExt <= targetP; pExt += stepP) {
    p.push_back(pExt);
    mean.push_back(lastMean);
    sigma.push_back(lastSigma);
  }
}

// Extend to LOW momentum (extrapolate from first valid points)
void extendToLowMomentum(std::vector<double>& p, std::vector<double>& mean, 
                          std::vector<double>& sigma, double targetP, 
                          double stepP = 5.0, int nAvg = 5) {
  if (p.empty()) return;
  double dataMinP = p.front();
  if (targetP >= dataMinP) return;
  
  int actualAvg = std::min(nAvg, (int)p.size());
  double firstMean = 0, firstSigma = 0;
  for (int i = 0; i < actualAvg; ++i) {
    firstMean += mean[i];
    firstSigma += sigma[i];
  }
  firstMean /= actualAvg;
  firstSigma /= actualAvg;
  
  // Insert at beginning
  std::vector<double> newP, newMean, newSigma;
  for (double pExt = targetP; pExt < dataMinP; pExt += stepP) {
    newP.push_back(pExt);
    newMean.push_back(firstMean);
    newSigma.push_back(firstSigma);
  }
  
  // Append original data
  for (size_t i = 0; i < p.size(); ++i) {
    newP.push_back(p[i]);
    newMean.push_back(mean[i]);
    newSigma.push_back(sigma[i]);
  }
  
  p = newP;
  mean = newMean;
  sigma = newSigma;
}

// =====================================================
// ROBUST FITTING FUNCTION - ELECTRON
// =====================================================

FitResult tryFitDeltaBeta(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx,
                          double pCenter, double pLow, double pHigh, double sliceWidth, int phaseTag,
                          const PropagatedParams& prevParams) {
  FitResult best;
  best.success = false;
  best.mean = 0.0;
  best.sigma = 0.01;
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
  best.bkgGausSigma = 0.05;

  if (!proj || best.entries < 30) return best;

  int bin_min = proj->FindFixBin(fitMin);
  int bin_max = proj->FindFixBin(fitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 10.0) return best;

  // STEP 1: Find peak
  int bin_sig_lo = proj->FindFixBin(-0.06);
  int bin_sig_hi = proj->FindFixBin(0.06);
  
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

  // Find 80% boundaries
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
  
  if (bin80_hi - bin80_lo < 2) {
    bin80_lo = std::max(bin_sig_lo, peakBin - 2);
    bin80_hi = std::min(bin_sig_hi, peakBin + 2);
    x80_lo = proj->GetBinCenter(bin80_lo);
    x80_hi = proj->GetBinCenter(bin80_hi);
  }

  // STEP 2: Preliminary Gaussian fit
  TF1* prelimFit = new TF1(Form("prelim_w%d_s%d", widthIdx, sliceIdx), "gaus", x80_lo, x80_hi);
  prelimFit->SetParameter(0, peakVal);
  prelimFit->SetParameter(1, peakPos);
  prelimFit->SetParameter(2, 0.01);
  prelimFit->SetParLimits(1, x80_lo, x80_hi);
  prelimFit->SetParLimits(2, 0.002, 0.05);
  
  TFitResultPtr prelimRes = proj->Fit(prelimFit, "SQR0B");
  
  double anchorMean = peakPos;
  double anchorSigma = 0.015;
  if (prelimRes.Get() && prelimRes->IsValid()) {
    anchorMean = prelimFit->GetParameter(1);
    anchorSigma = prelimFit->GetParameter(2);
  }
  delete prelimFit;

  // Background estimate
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

  double initSigMean = anchorMean;
  double initSigSigma = anchorSigma;
  double initSigAmp = sigAmpEst;
  
  if (prevParams.valid) {
    initSigSigma = prevParams.sigSigma;
    double ampScale = integral / 1000.0;
    initSigAmp = prevParams.sigAmp * ampScale;
    if (initSigAmp < 0.5) initSigAmp = sigAmpEst;
    if (initSigAmp > 10.0 * peakVal) initSigAmp = sigAmpEst;
  }

  // STEP 3: Full fit with all components
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
    
    TString funcExpr = "gaus(0) + gaus(3)";
    for (int pp = 0; pp <= polyOrder; ++pp) {
      if (pp == 0) funcExpr += Form(" + [%d]", 6 + pp);
      else funcExpr += Form(" + [%d]*pow(x,%d)", 6 + pp, pp);
    }
    
    TString funcName = Form("fullfit_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fullFit = new TF1(funcName, funcExpr, fitMin, fitMax);
    
    fullFit->SetParameter(0, initSigAmp);
    fullFit->SetParameter(1, anchorMean);
    fullFit->SetParameter(2, initSigSigma);
    fullFit->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal));
    double meanTol = 0.01;
    fullFit->SetParLimits(1, anchorMean - meanTol, anchorMean + meanTol);
    fullFit->SetParLimits(2, 0.003, 0.06);
    
    double edgeMax = std::max(bkgLeft, bkgRight);
    double maxBkgAmp = std::min(0.3 * initSigAmp, 2.0 * edgeMax);
    if (maxBkgAmp < 0.01 * initSigAmp) maxBkgAmp = 0.1 * initSigAmp;
    
    fullFit->SetParameter(3, 0.05 * initSigAmp);
    fullFit->SetParLimits(3, 0.0, maxBkgAmp);
    fullFit->SetParameter(4, 0.06);
    fullFit->SetParLimits(4, 0.03, 0.12);
    fullFit->SetParameter(5, 0.05);
    fullFit->SetParLimits(5, 0.025, 0.10);
    
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
    
    if (chi2ndf > 200.0) continue;
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

  // Select best model
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
  
  if (bestIdx < 0) return best;

  // STEP 4 & 5: Subtract background and refit
  FullFitResult& chosen = results[bestIdx];
  
  TH1D* hSubtracted = (TH1D*)proj->Clone(Form("hsub_w%d_s%d", widthIdx, sliceIdx));
  
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
  
  TF1* bkgGausFunc = new TF1("bkgGausFunc", "gaus", fitMin, fitMax);
  bkgGausFunc->SetParameters(chosen.bkgAmp, chosen.bkgMean, chosen.bkgSigma);
  
  for (int b = 1; b <= hSubtracted->GetNbinsX(); ++b) {
    double x = hSubtracted->GetBinCenter(b);
    double origVal = hSubtracted->GetBinContent(b);
    double polyVal = polyFunc->Eval(x);
    double bkgGausVal = bkgGausFunc->Eval(x);
    double newVal = origVal - polyVal - bkgGausVal;
    hSubtracted->SetBinContent(b, newVal);
  }
  
  delete polyFunc;
  delete bkgGausFunc;
  
  // Final Gaussian fit
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
// MAIN FUNCTION - ELECTRON PID
// =====================================================

void pid_em_cut_pp45_sim() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain for ELECTRON data ---
  const char* treeName = "Em_ID";
  
  // Input file - opened twice (second time as "fake" systematics)
  const std::vector<TString> files = {
    "/hdd2/przygoda/hades/pp45/049/lepton_sim_pid.root"
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
    cout << "No entries found in TChain." << endl;
    return;
  }
  cout << "TChain: " << added << " files, " << nEnt << " entries" << endl;

  // =====================================================
  // CREATE 2D HISTOGRAMS - ELECTRON
  // =====================================================
  
  // Main histogram: p vs Δβ (electron)
  const char* h2name = "h2_em_deltaBeta";
  TString drawCmd = Form(
    "em_beta - em_p/sqrt(em_p*em_p + %.4f*%.4f) : em_p >> %s(400,0,1600,300,-0.15,0.15)",
    gElectronMass, gElectronMass, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "eVertReco_z>-500 && em_p>0 && em_beta>0 && em_beta<1.5 && em_sim_id==3", "colz");
  TH2F* h2DB = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2DB) {
    cout << "Failed to create Δβ histogram" << endl;
    return;
  }

  h2DB->SetTitle("Electron: Momentum vs #Delta#beta (#beta - #beta_{e^{-}})");
  h2DB->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DB->GetYaxis()->SetTitle("#Delta#beta = #beta - #beta_{e^{-}}");

  // (p, β) histogram
  const char* h2pb_name = "h2_em_beta";
  if (gDirectory->FindObject(h2pb_name)) gDirectory->Delete(Form("%s;*", h2pb_name));
  chain->Draw(Form("em_beta : em_p >> %s(400,0,1600,300,0.0,1.2)", h2pb_name), 
              "eVertReco_z>-500 && em_p>0 && em_beta>0 && em_beta<1.5 && em_sim_id==3", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2pb_name));
  if (h2PB) {
    h2PB->SetTitle("Electron: #beta vs Momentum");
    h2PB->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2PB->GetYaxis()->SetTitle("#beta");
  }

  // (mass, a) histogram - SAME FORMULA AS ORIGINAL
  // X-axis: mass = p*sqrt(1/β² - 1)
  // Y-axis: a = sqrt(1 + s²p² - (1-β²)²)
  const char* h2ma_name = "h2_em_mass_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*em_p*em_p - pow(1-em_beta*em_beta,2)) : "
    "em_p*sqrt(1/pow(em_beta,2) - 1) >> %s(300,0,300,160,0,16)",
    gSSquared, h2ma_name);
  if (gDirectory->FindObject(h2ma_name)) gDirectory->Delete(Form("%s;*", h2ma_name));
  chain->Draw(drawMA, "eVertReco_z>-500 && em_p>0 && em_beta>0 && em_beta<1.5 && em_sim_id==3", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2ma_name));
  if (h2MA) {
    h2MA->SetTitle("Electron: a-parameter vs Mass");
    h2MA->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
    h2MA->GetYaxis()->SetTitle("a");
  }

  // (p, mass²) histogram - SAME FORMAT AS ORIGINAL
  // X-axis: momentum
  // Y-axis: mass² = p²(1/β² - 1)
  const char* h2m2_name = "h2_em_mass2";
  if (gDirectory->FindObject(h2m2_name)) gDirectory->Delete(Form("%s;*", h2m2_name));
  chain->Draw(Form("em_p*em_p*(1.0/(em_beta*em_beta) - 1) : em_p >> %s(400,0,1600,400,-20000,60000)", h2m2_name), 
              "eVertReco_z>-500 && em_p>0 && em_beta>0.1 && em_beta<1.5 && em_sim_id==3", "colz");
  TH2F* h2M2 = static_cast<TH2F*>(gDirectory->Get(h2m2_name));
  if (h2M2) {
    h2M2->SetTitle("Electron: Mass^{2} vs Momentum");
    h2M2->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2M2->GetYaxis()->SetTitle("Mass^{2} [MeV^{2}/c^{4}]");
  }

  // =====================================================
  // SCANNING PARAMETERS - ELECTRON
  // Same slice widths as pion: 10, 20, 30, 40 MeV/c
  // =====================================================
  
  const double warmupLow = 130.0;       // Warmup fit range [130, 160] MeV/c
  const double warmupHigh = 160.0;
  const double startMom = 80.0;         // Anchor point
  const double transitionMom = 400.0;   // Where Phase 2 doubling starts
  const double endMom = 1200.0;         // End of fitting (extrapolate beyond)
  const double stepSize = 1.0;          // 1 MeV/c steps in Phase 1
  
  const int nWidths = 4;
  const double baseWidths[nWidths] = {10.0, 20.0, 30.0, 40.0};  // SAME AS PION
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

    std::vector<double> momLows, momHighs, momCenters, sliceWidths;
    std::vector<int> phaseTag;

    // Phase 0: RIGHT EDGE anchored at startMom, width DOUBLES going left
    std::vector<int> phase0Indices;
    {
      double pRight = startMom;
      double currentWidth = baseW;
      double pLeft = pRight - currentWidth;
      
      while (pLeft > 0) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(0);
        phase0Indices.push_back(idx);
        
        currentWidth *= 2.0;
        pLeft = pRight - currentWidth;
      }
      
      int idx = momLows.size();
      momLows.push_back(0);
      momHighs.push_back(pRight);
      momCenters.push_back(0.5 * pRight);
      sliceWidths.push_back(pRight);
      phaseTag.push_back(0);
      phase0Indices.push_back(idx);
    }
    int nPhase0 = phase0Indices.size();

    // Phase 1: forward sliding slices
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

    // Phase 2: Progressive doubling with sliding
    std::vector<int> phase2Indices;
    {
      double pLeft = transitionMom;
      double prevWidth = baseW;
      double currentWidth = baseW * 2.0;
      
      while (pLeft + currentWidth <= endMom) {
        int nSteps = static_cast<int>(prevWidth);
        
        for (int step = 0; step <= nSteps && pLeft + currentWidth <= endMom; ++step) {
          int idx = momLows.size();
          momLows.push_back(pLeft);
          momHighs.push_back(pLeft + currentWidth);
          momCenters.push_back(pLeft + 0.5 * currentWidth);
          sliceWidths.push_back(currentWidth);
          phaseTag.push_back(2);
          phase2Indices.push_back(idx);
          
          if (step < nSteps) pLeft += stepSize;
        }
        
        prevWidth = currentWidth;
        currentWidth *= 2.0;
      }
      
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
    
    cout << "Phase 0 slices (right edge at " << startMom << ", width doubling):" << endl;
    for (size_t k = 0; k < phase0Indices.size(); ++k) {
      int idx = phase0Indices[k];
      cout << Form("  [%3.0f, %3.0f] width=%3.0f", 
                   momLows[idx], momHighs[idx], sliceWidths[idx]) << endl;
    }
    
    cout << "Slices: Phase0=" << nPhase0 << ", Phase1=" << nPhase1 
         << ", Phase2=" << nPhase2 << ", Total=" << nSlices << endl;

    allMomCenters[w] = momCenters;
    allFitResults[w].resize(nSlices);

    // =====================================================
    // FIT IN CORRECT ORDER
    // =====================================================
    
    int nSuccess = 0;
    PropagatedParams warmupParams;
    warmupParams.valid = false;
    PropagatedParams anchorParams;
    anchorParams.valid = false;
    
    // WARMUP FIT
    cout << "Fitting WARMUP [" << warmupLow << "," << warmupHigh << "]..." << endl;
    {
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(warmupLow + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(warmupHigh - 0.01);
      
      TString projName = Form("proj_db_w%d_warmup", w);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      
      PropagatedParams noParams;
      noParams.valid = false;
      
      double warmupCenter = 0.5 * (warmupLow + warmupHigh);
      double warmupWidth = warmupHigh - warmupLow;
      
      FitResult warmupResult = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, -1, w,
                                                warmupCenter, warmupLow, warmupHigh, 
                                                warmupWidth, -1, noParams);
      delete proj;
      
      if (warmupResult.success) {
        warmupParams = fitResultToProps(warmupResult);
        cout << Form("  WARMUP OK: μ=%.4f, σ=%.4f, χ²=%.2f", 
                     warmupResult.mean, warmupResult.sigma, warmupResult.chi2ndf) << endl;
      } else {
        cout << "  WARMUP FAILED!" << endl;
      }
    }
    
    // ANCHOR FIT
    cout << "Fitting ANCHOR at p=" << startMom << "..." << endl;
    if (phase1Indices.size() > 0) {
      int anchorIdx = phase1Indices[0];
      
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[anchorIdx] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[anchorIdx] - 0.01);
      
      TString projName = Form("proj_db_w%d_anchor", w);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      
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
        cout << "  ANCHOR FAILED!" << endl;
      }
    }
    
    // Phase 0
    cout << "Fitting Phase 0..." << endl;
    PropagatedParams prevParams = warmupParams;
    
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
      }
      
      delete proj;
    }
    
    // Phase 1
    cout << "Fitting Phase 1..." << endl;
    prevParams = anchorParams;
    
    for (size_t k = 1; k < phase1Indices.size(); ++k) {
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
    
    PropagatedParams phase1EndParams = prevParams;
    
    // Phase 2
    cout << "Fitting Phase 2..." << endl;
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
      }
      
      delete proj;
    }
    
    cout << "Total success: " << nSuccess << "/" << nSlices << endl;

    // =====================================================
    // SELECT DISPLAY INDICES
    // =====================================================
    std::vector<int> displayIndices;
    
    if (nPhase0 > 0) {
      displayIndices.push_back(phase0Indices[0]);
      if (nPhase0 > 2) displayIndices.push_back(phase0Indices[nPhase0/2]);
      displayIndices.push_back(phase0Indices[nPhase0-1]);
    }
    
    std::vector<double> targetMomenta = {100, 150, 200, 300, 400, 500, 700, 900, 1100};
    
    for (double target : targetMomenta) {
      int bestIdx = -1;
      double bestDist = 1e9;
      
      for (size_t k = 0; k < phase1Indices.size(); ++k) {
        int idx = phase1Indices[k];
        double dist = std::abs(momCenters[idx] - target);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      }
      
      for (size_t k = 0; k < phase2Indices.size(); ++k) {
        int idx = phase2Indices[k];
        double dist = std::abs(momCenters[idx] - target);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      }
      
      if (bestIdx >= 0 && bestDist < 100) {
        displayIndices.push_back(bestIdx);
      }
    }
    
    std::sort(displayIndices.begin(), displayIndices.end());
    displayIndices.erase(std::unique(displayIndices.begin(), displayIndices.end()), displayIndices.end());
    if (displayIndices.size() > (size_t)nDisplayPerWidth)
      displayIndices.resize(nDisplayPerWidth);

    cFits[w] = new TCanvas(Form("c_fits_dBeta_em_%d", w), 
                           Form("Delta-Beta Fits (Electron e^{-}) - %s", widthLabels[w].Data()), 1400, 900);
    cFits[w]->Divide(4, 3);

    // =====================================================
    // DISPLAY FITS - SHOW ALL COMPONENTS
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
        
        // Total fit: gaus(0) + gaus(3) + poly at [6]
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

        // Signal Gaussian (blue solid)
        TF1* sig = new TF1(Form("sig_db_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
        sig->SetParameters(result.amplitude, result.mean, result.sigma);
        sig->SetLineColor(kBlue);
        sig->SetLineStyle(1);
        sig->SetLineWidth(3);
        sig->Draw("same");

        // Background Gaussian (orange dashed)
        if (result.bkgGausAmp > 0.001 * result.amplitude) {
          TF1* bkgGaus = new TF1(Form("bkg_db_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
          bkgGaus->SetParameters(result.bkgGausAmp, result.bkgGausMean, result.bkgGausSigma);
          bkgGaus->SetLineColor(kOrange-1);
          bkgGaus->SetLineStyle(kDashed);
          bkgGaus->SetLineWidth(2);
          bkgGaus->Draw("same");
        }

        // Polynomial (green dashed)
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

      // Labels
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
        tex.SetTextColor(kOrange-1);
        tex.DrawLatex(0.55, 0.65, Form("bkg=%.0f%%", 100.0 * result.bkgGausAmp / result.amplitude));
        tex.SetTextColor(kBlack);
        tex.DrawLatex(0.55, 0.58, Form("pol%d", result.polyOrder));
        if (result.chi2ndf >= 0.3 && result.chi2ndf <= 4.0) {
          tex.SetTextColor(kGreen+2);
        } else {
          tex.SetTextColor(kOrange-1);
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
    cFits[w]->SaveAs(Form("em_fits_%s.png", widthLabels[w].Data()));
  }

  // =====================================================
  // CONTOUR PLOTS IN Δβ SPACE
  // =====================================================
  
  TCanvas* cCombDB_1sig = new TCanvas("c_comb_db_1sig_em", "Combined 1#sigma in #Delta#beta (Electron)", 1000, 800);
  cCombDB_1sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB1 = new TLine(0, 0, gExtendTo, 0);
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    // Extend to HIGH momentum
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    // Extend to LOW momentum
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
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
  legDB_1->AddEntry(zeroLineDB1, "#Delta#beta=0 (e^{-})", "l");
  legDB_1->Draw();
  cCombDB_1sig->Modified();
  cCombDB_1sig->Update();
  cCombDB_1sig->SaveAs("em_contours_db_1sig.png");

  // Combined 3σ in Δβ
  TCanvas* cCombDB_3sig = new TCanvas("c_comb_db_3sig_em", "Combined 3#sigma in #Delta#beta (Electron)", 1000, 800);
  cCombDB_3sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB3 = new TLine(0, 0, gExtendTo, 0);
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
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
  legDB_3->AddEntry(zeroLineDB3, "#Delta#beta=0 (e^{-})", "l");
  legDB_3->Draw();
  cCombDB_3sig->Modified();
  cCombDB_3sig->Update();
  cCombDB_3sig->SaveAs("em_contours_db_3sig.png");

  // =====================================================
  // CONTOUR PLOTS IN (p, β) SPACE
  // =====================================================
  
  // Combined 1σ in (p, β)
  TCanvas* cCombPB_1sig = new TCanvas("c_comb_pb_1sig_em", "Combined 1#sigma in (p, #beta) (Electron)", 1000, 800);
  cCombPB_1sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* emCurvePB1 = new TF1("emCurvePB1", Form("x/sqrt(x*x + %f*%f)", gElectronMass, gElectronMass), 0, gExtendTo);
  emCurvePB1->SetLineColor(kBlack);
  emCurvePB1->SetLineStyle(kDashed);
  emCurvePB1->SetLineWidth(2);
  emCurvePB1->Draw("same");
  
  TLegend* legPB_1 = new TLegend(0.60, 0.15, 0.88, 0.40);
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> pPoints1Upper, betaPoints1Upper;
    std::vector<double> pPoints1Lower, betaPoints1Lower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 1.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints1Upper.push_back(p);
        betaPoints1Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_em + smoothMean[i] - 1.0 * smoothSigma[i];
      if (beta_lower > 0.1 && beta_lower < 1.15) {
        pPoints1Lower.push_back(p);
        betaPoints1Lower.push_back(beta_lower);
      }
    }

    if (!pPoints1Upper.empty()) {
      TGraph* gUpper = new TGraph(pPoints1Upper.size(), pPoints1Upper.data(), betaPoints1Upper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(3);
      gUpper->SetLineStyle(kDashed);
      gUpper->Draw("L");
      legPB_1->AddEntry(gUpper, Form("%s", widthLabels[w].Data()), "l");
    }
    if (!pPoints1Lower.empty()) {
      TGraph* gLower = new TGraph(pPoints1Lower.size(), pPoints1Lower.data(), betaPoints1Lower.data());
      gLower->SetLineColor(widthColors[w]);
      gLower->SetLineWidth(3);
      gLower->SetLineStyle(kDashed);
      gLower->Draw("L");
    }
  }
  legPB_1->AddEntry(emCurvePB1, "m_{e}=0.511", "l");
  legPB_1->Draw();
  cCombPB_1sig->Modified();
  cCombPB_1sig->Update();
  cCombPB_1sig->SaveAs("em_contours_pb_1sig.png");

  // Combined 3σ in (p, β)
  TCanvas* cCombPB_3sig = new TCanvas("c_comb_pb_3sig_em", "Combined 3#sigma in (p, #beta) (Electron)", 1000, 800);
  cCombPB_3sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  // Electron β(p) theory curve
  TF1* emCurvePB3 = new TF1("emCurvePB3", Form("x/sqrt(x*x + %f*%f)", gElectronMass, gElectronMass), 0, gExtendTo);
  emCurvePB3->SetLineColor(kBlack);
  emCurvePB3->SetLineStyle(kDashed);
  emCurvePB3->SetLineWidth(2);
  emCurvePB3->Draw("same");
  
  TLegend* legPB_3 = new TLegend(0.60, 0.15, 0.88, 0.40);
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> pPoints3Upper, betaPoints3Upper;
    std::vector<double> pPoints3Lower, betaPoints3Lower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 3.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints3Upper.push_back(p);
        betaPoints3Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_em + smoothMean[i] - 3.0 * smoothSigma[i];
      if (beta_lower > 0.1 && beta_lower < 1.15) {
        pPoints3Lower.push_back(p);
        betaPoints3Lower.push_back(beta_lower);
      }
    }

    if (!pPoints3Upper.empty()) {
      TGraph* gUpper = new TGraph(pPoints3Upper.size(), pPoints3Upper.data(), betaPoints3Upper.data());
      gUpper->SetLineColor(widthColors[w]);
      gUpper->SetLineWidth(3);
      gUpper->SetLineStyle(kDashed);
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
  }
  legPB_3->AddEntry(emCurvePB3, "m_{e}=0.511", "l");
  legPB_3->Draw();
  cCombPB_3sig->Modified();
  cCombPB_3sig->Update();
  cCombPB_3sig->SaveAs("em_contours_pb_3sig.png");

  // =====================================================
  // CONTOUR PLOTS IN (p, mass²) SPACE
  // =====================================================
  
  auto mass2FromPBeta = [](double p, double beta) -> double {
    if (beta <= 0.01 || beta >= 1.5) return -999999;
    return p * p * (1.0 / (beta * beta) - 1.0);
  };
  
  const double emMass2 = gElectronMass * gElectronMass;  // ~0.26 MeV²/c⁴
  
  // Combined 1σ in (p, mass²)
  TCanvas* cCombM2_1sig = new TCanvas("c_comb_m2_1sig_em", "Combined 1#sigma in (p, mass^{2}) (Electron)", 1000, 800);
  cCombM2_1sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  TLine* emLineM2_1 = new TLine(0, emMass2, gExtendTo, emMass2);
  emLineM2_1->SetLineColor(kBlack);
  emLineM2_1->SetLineStyle(kDashed);
  emLineM2_1->SetLineWidth(2);
  emLineM2_1->Draw("same");
  
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> pPointsUpper, m2PointsUpper;
    std::vector<double> pPointsLower, m2PointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 1.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > -50000 && m2_upper < 100000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_em + smoothMean[i] - 1.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > -50000 && m2_lower < 100000) {
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
  legM2_1->AddEntry(emLineM2_1, "m_{e}^{2}=0.26", "l");
  legM2_1->Draw();
  cCombM2_1sig->Modified();
  cCombM2_1sig->Update();
  cCombM2_1sig->SaveAs("em_contours_m2_1sig.png");

  // Combined 3σ in (p, mass²)
  TCanvas* cCombM2_3sig = new TCanvas("c_comb_m2_3sig_em", "Combined 3#sigma in (p, mass^{2}) (Electron)", 1000, 800);
  cCombM2_3sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  TLine* emLineM2 = new TLine(0, emMass2, gExtendTo, emMass2);
  emLineM2->SetLineColor(kBlack);
  emLineM2->SetLineStyle(kDashed);
  emLineM2->SetLineWidth(2);
  emLineM2->Draw("same");
  
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> pPointsUpper, m2PointsUpper;
    std::vector<double> pPointsLower, m2PointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 3.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > -50000 && m2_upper < 100000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_em + smoothMean[i] - 3.0 * smoothSigma[i];
      double m2_lower = mass2FromPBeta(p, beta_lower);
      if (m2_lower > -50000 && m2_lower < 100000) {
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
  legM2_3->AddEntry(emLineM2, "m_{e}^{2}=0.26", "l");
  legM2_3->Draw();
  cCombM2_3sig->Modified();
  cCombM2_3sig->Update();
  cCombM2_3sig->SaveAs("em_contours_m2_3sig.png");

  // =====================================================
  // CONTOUR PLOTS IN (mass, a) SPACE
  // =====================================================
  
  // Combined 1σ in (mass, a)
  TCanvas* cCombMA_1sig = new TCanvas("c_comb_ma_1sig_em", "Combined 1#sigma in (mass, a) (Electron)", 1000, 800);
  cCombMA_1sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* emLineMA_1 = new TLine(gElectronMass, 0, gElectronMass, 16);
  emLineMA_1->SetLineColor(kBlack);
  emLineMA_1->SetLineStyle(kDashed);
  emLineMA_1->SetLineWidth(2);
  emLineMA_1->Draw("same");
  
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> massPointsUpper, aPointsUpper;
    std::vector<double> massPointsLower, aPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 1.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > -50 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_em + smoothMean[i] - 1.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > -50 && mass_l < 300 && a_l > 0 && a_l < 16) {
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
  legMA_1->AddEntry(emLineMA_1, "m_{e}=0.511", "l");
  legMA_1->Draw();
  cCombMA_1sig->Modified();
  cCombMA_1sig->Update();
  cCombMA_1sig->SaveAs("em_contours_ma_1sig.png");

  // Combined 3σ in (mass, a)
  TCanvas* cCombMA_3sig = new TCanvas("c_comb_ma_3sig_em", "Combined 3#sigma in (mass, a) (Electron)", 1000, 800);
  cCombMA_3sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* emLineMA = new TLine(gElectronMass, 0, gElectronMass, 16);
  emLineMA->SetLineColor(kBlack);
  emLineMA->SetLineStyle(kDashed);
  emLineMA->SetLineWidth(2);
  emLineMA->Draw("same");
  
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
    
    sortByMomentum(rawP, rawMean, rawSigma);
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 20.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    std::vector<double> massPointsUpper, aPointsUpper;
    std::vector<double> massPointsLower, aPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_em = betaElectron(p);
      
      double beta_upper = beta_em + smoothMean[i] + 3.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > -50 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_em + smoothMean[i] - 3.0 * smoothSigma[i];
      double mass_l, a_l;
      if (beta_lower > 0.1 && beta_lower < 1.5) {
        if (pBetaToMassA(p, beta_lower, gSSquared, mass_l, a_l) && 
            mass_l > -50 && mass_l < 300 && a_l > 0 && a_l < 16) {
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
  legMA_3->AddEntry(emLineMA, "m_{e}=0.511", "l");
  legMA_3->Draw();
  cCombMA_3sig->Modified();
  cCombMA_3sig->Update();
  cCombMA_3sig->SaveAs("em_contours_ma_3sig.png");

  // =====================================================
  // PARAMETER PLOTS
  // =====================================================
  
  TCanvas* cPar = new TCanvas("c_par_em", "Fit Parameters vs Momentum (Electron)", 1200, 800);
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
  TLine* zeroMean = new TLine(0, 0, gExtendTo, 0);
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
  mgSigma->SetTitle("Sigma #Delta#beta vs Momentum;p [MeV/c];#sigma (#Delta#beta)");
  mgSigma->Draw("A");
  mgSigma->GetYaxis()->SetRangeUser(0.0, 0.05);
  legSigma->Draw();

  // Chi2
  cPar->cd(3);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgChi2 = new TMultiGraph();
  TLegend* legChi2 = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success && allFitResults[w][i].chi2ndf < 20) {
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
  mgChi2->GetYaxis()->SetRangeUser(0.0, 10.0);
  TLine* chi2Line1 = new TLine(0, 1, gExtendTo, 1);
  chi2Line1->SetLineColor(kGreen+2);
  chi2Line1->SetLineStyle(kDashed);
  chi2Line1->Draw("same");
  legChi2->Draw();

  // Entries
  cPar->cd(4);
  gPad->SetLeftMargin(0.14);
  gPad->SetLogy();
  TMultiGraph* mgEnt = new TMultiGraph();
  TLegend* legEnt = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].entries);
      }
    }
    g->SetMarkerStyle(20);
    g->SetMarkerSize(0.3);
    g->SetMarkerColor(widthColors[w]);
    g->SetLineColor(widthColors[w]);
    mgEnt->Add(g, "P");
    legEnt->AddEntry(g, widthLabels[w], "p");
  }
  mgEnt->SetTitle("Entries vs Momentum;p [MeV/c];Entries");
  mgEnt->Draw("A");
  legEnt->Draw();

  cPar->Modified();
  cPar->Update();
  cPar->SaveAs("em_parameters.png");

  // =====================================================
  // GENERATE TCutG AND FINAL VISUALIZATION
  // =====================================================
  
  // Use width index 1 (20 MeV/c) as default - best balance
  int selectedWidth = 1;
  
  std::vector<double> rawP, rawMean, rawSigma;
  for (size_t i = 0; i < allFitResults[selectedWidth].size(); ++i) {
    if (allFitResults[selectedWidth][i].success) {
      rawP.push_back(allMomCenters[selectedWidth][i]);
      rawMean.push_back(allFitResults[selectedWidth][i].mean);
      rawSigma.push_back(allFitResults[selectedWidth][i].sigma);
    }
  }
  
  // TCutG objects for visualization and saving
  TCutG* cut1sig = nullptr;
  TCutG* cut25sig = nullptr;
  TCutG* cut3sig = nullptr;
  TCutG* cut35sig = nullptr;
  TCutG* cut5sig = nullptr;
  
  TGraph* meanGraph = nullptr;
  TGraph* sigmaGraph = nullptr;
  TSpline3* meanSpline = nullptr;
  TSpline3* sigmaSpline = nullptr;
  
  if (rawP.size() >= 5) {
    sortByMomentum(rawP, rawMean, rawSigma);
    
    std::vector<double> smoothMean = robustSmooth(rawMean, 7, 9);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, 7, 9);
    
    // Extend to high and low momentum
    extendToMomentum(rawP, smoothMean, smoothSigma, gExtendTo, 10.0, 5);
    extendToLowMomentum(rawP, smoothMean, smoothSigma, gExtendLow, 5.0, 5);
    
    // Create TGraphs and TSplines for smoothed parameters
    meanGraph = new TGraph(rawP.size(), rawP.data(), smoothMean.data());
    meanGraph->SetName("gr_mean_deltabeta");
    meanGraph->SetTitle("Smoothed Mean #Delta#beta vs p");
    
    sigmaGraph = new TGraph(rawP.size(), rawP.data(), smoothSigma.data());
    sigmaGraph->SetName("gr_sigma_deltabeta");
    sigmaGraph->SetTitle("Smoothed Sigma #Delta#beta vs p");
    
    meanSpline = new TSpline3("spl_mean_deltabeta", meanGraph);
    sigmaSpline = new TSpline3("spl_sigma_deltabeta", sigmaGraph);
    
    // Lambda to generate TCutG for a given sigma level
    auto makeCut = [&](double nSig, const char* cutName) -> TCutG* {
      std::vector<double> pCut, betaCut;
      
      // Upper boundary (forward)
      for (size_t i = 0; i < rawP.size(); ++i) {
        double p = rawP[i];
        if (p < 0) continue;
        double beta_em = betaElectron(p);
        double beta_upper = beta_em + smoothMean[i] + nSig * smoothSigma[i];
        if (beta_upper > 0.01 && beta_upper < 1.5) {
          pCut.push_back(p);
          betaCut.push_back(beta_upper);
        }
      }
      
      // Lower boundary (backward)
      for (int i = rawP.size() - 1; i >= 0; --i) {
        double p = rawP[i];
        if (p < 0) continue;
        double beta_em = betaElectron(p);
        double beta_lower = beta_em + smoothMean[i] - nSig * smoothSigma[i];
        if (beta_lower > 0.01 && beta_lower < 1.5) {
          pCut.push_back(p);
          betaCut.push_back(beta_lower);
        }
      }
      
      // Close polygon
      if (!pCut.empty()) {
        pCut.push_back(pCut[0]);
        betaCut.push_back(betaCut[0]);
      }
      
      TCutG* cut = new TCutG(cutName, pCut.size(), pCut.data(), betaCut.data());
      cut->SetVarX("em_p");
      cut->SetVarY("em_beta");
      return cut;
    };
    
    // Generate all TCutG objects
    cut1sig  = makeCut(1.0, "cut_em_1sig_w1");
    cut25sig = makeCut(2.5, "cut_em_25sig_w1");
    cut3sig  = makeCut(3.0, "cut_em_3sig_w1");
    cut35sig = makeCut(3.5, "cut_em_35sig_w1");
    cut5sig  = makeCut(5.0, "cut_em_5sig_w1");
    
    // Set visual properties for display (1σ, 3σ, 5σ)
    cut1sig->SetLineColor(kRed);
    cut1sig->SetLineWidth(3);
    cut1sig->SetLineStyle(kSolid);
    
    cut3sig->SetLineColor(kBlue);
    cut3sig->SetLineWidth(3);
    cut3sig->SetLineStyle(kDashed);
    
    cut5sig->SetLineColor(kGreen+2);
    cut5sig->SetLineWidth(3);
    cut5sig->SetLineStyle(7);  // Long dashed
    
    cout << "[TCutG] Generated cuts: " << cut1sig->GetN() << " points each" << endl;
  }
  
  // =====================================================
  // FINAL CANVAS: TCutG ON (p, β) HISTOGRAM
  // Display: 1σ, 3σ, 5σ
  // =====================================================
  
  TCanvas* cCuts = new TCanvas("c_em_cuts", "Electron PID Cuts in (p, #beta)", 1000, 800);
  cCuts->cd();
  
  if (h2PB) h2PB->Draw("colz");
  
  // Theory curve: β = p/√(p² + m²)
  TF1* theoryCurve = new TF1("theoryCurve_em", 
    Form("x/sqrt(x*x + %f*%f)", gElectronMass, gElectronMass), 0, gExtendTo);
  theoryCurve->SetLineColor(kBlack);
  theoryCurve->SetLineStyle(kDashed);
  theoryCurve->SetLineWidth(2);
  theoryCurve->Draw("same");
  
  // Draw TCutG (1σ, 3σ, 5σ)
  if (cut1sig) cut1sig->Draw("L same");
  if (cut3sig) cut3sig->Draw("L same");
  if (cut5sig) cut5sig->Draw("L same");
  
  TLegend* legCuts = new TLegend(0.55, 0.15, 0.88, 0.40);
  legCuts->SetHeader(Form("Electron e^{-} (width %s)", widthLabels[selectedWidth].Data()));
  if (cut1sig) legCuts->AddEntry(cut1sig, "1#sigma", "l");
  if (cut3sig) legCuts->AddEntry(cut3sig, "3#sigma", "l");
  if (cut5sig) legCuts->AddEntry(cut5sig, "5#sigma", "l");
  legCuts->AddEntry(theoryCurve, Form("m_{e}=%.3f MeV", gElectronMass), "l");
  legCuts->Draw();
  
  cCuts->Modified();
  cCuts->Update();
  cCuts->SaveAs("em_pid_cuts_overlay.png");
  
  // =====================================================
  // SAVE TO ROOT FILE
  // Store: 1σ, 2.5σ, 3σ, 3.5σ, 5σ
  // =====================================================
  
  TFile* fOut = new TFile("em_pid_cuts_pp45_sim.root", "RECREATE");
  
  // Save raw fit results as TTree
  TTree* tree = new TTree("FitResults", "Raw PID fit results in DeltaBeta representation");
  
  Double_t t_p, t_pLow, t_pHigh, t_mean, t_sigma, t_chi2, t_sliceWidth;
  Int_t t_widthIdx, t_phaseTag, t_success;
  
  tree->Branch("p_center",    &t_p);
  tree->Branch("p_low",       &t_pLow);
  tree->Branch("p_high",      &t_pHigh);
  tree->Branch("delta_beta_mean",  &t_mean);
  tree->Branch("delta_beta_sigma", &t_sigma);
  tree->Branch("chi2ndf",     &t_chi2);
  tree->Branch("slice_width", &t_sliceWidth);
  tree->Branch("width_idx",   &t_widthIdx);
  tree->Branch("phase_tag",   &t_phaseTag);
  tree->Branch("success",     &t_success);
  
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allFitResults[w].size(); ++i) {
      const FitResult& r = allFitResults[w][i];
      t_p = r.pCenter;
      t_pLow = r.pLow;
      t_pHigh = r.pHigh;
      t_mean = r.mean;
      t_sigma = r.sigma;
      t_chi2 = r.chi2ndf;
      t_sliceWidth = r.sliceWidth;
      t_widthIdx = w;
      t_phaseTag = r.phaseTag;
      t_success = r.success ? 1 : 0;
      tree->Fill();
    }
  }
  tree->Write();
  cout << "  - TTree 'FitResults' with " << tree->GetEntries() << " entries" << endl;
  
  // Save contour graphs and splines
  if (meanGraph && sigmaGraph && meanSpline && sigmaSpline) {
    TDirectory* dirContours = fOut->mkdir("Contours");
    dirContours->cd();
    
    meanGraph->Write();
    sigmaGraph->Write();
    meanSpline->Write();
    sigmaSpline->Write();
    
    cout << "  - Smoothed TGraphs and TSpline3 objects" << endl;
  }
  
  // Save TCutG objects (ALL 5 levels)
  if (cut1sig && cut25sig && cut3sig && cut35sig && cut5sig) {
    TDirectory* dirCuts = fOut->mkdir("TCutG");
    dirCuts->cd();
    
    cut1sig->Write();
    cut25sig->Write();
    cut3sig->Write();
    cut35sig->Write();
    cut5sig->Write();
    
    cout << "  - TCutG objects: cut_em_1sig, cut_em_25sig, cut_em_3sig, cut_em_35sig, cut_em_5sig" << endl;
  }
  
  // Save metadata
  fOut->cd();
  TNamed* metaParticle = new TNamed("particle", "electron");
  TNamed* metaMass = new TNamed("mass_MeV", Form("%.3f", gElectronMass));
  TNamed* metaWidth = new TNamed("selected_width", Form("%d", selectedWidth));
  TNamed* metaWidthLabel = new TNamed("width_label", widthLabels[selectedWidth].Data());
  metaParticle->Write();
  metaMass->Write();
  metaWidth->Write();
  metaWidthLabel->Write();
  
  fOut->Close();
  delete fOut;
  
  cout << "[TCutG] Output file 'em_pid_cuts_pp45_sim.root' saved successfully." << endl;

  // =====================================================
  // SUMMARY
  // =====================================================
  cout << "\n=========================================" << endl;
  cout << "=== ELECTRON PID ANALYSIS COMPLETE ===" << endl;
  cout << "=========================================" << endl;
  cout << "Particle: e- (electron), mass = " << gElectronMass << " MeV/c²" << endl;
  cout << "Slice widths: 10, 20, 30, 40 MeV/c (same as pion)" << endl;
  cout << "Warmup range: [" << warmupLow << ", " << warmupHigh << "] MeV/c" << endl;
  cout << "TCutG extended from p=" << gExtendLow << " to p=" << gExtendTo << " MeV/c" << endl;
  cout << "\nROBUST FIT MODEL:" << endl;
  cout << "  1. Find peak max and 80% boundaries" << endl;
  cout << "  2. Preliminary Gauss fit to peak top (anchors position)" << endl;
  cout << "  3. Full fit: Signal Gauss + Bkg Gauss + Polynomial" << endl;
  cout << "  4. Subtract polynomial and bkg Gauss from data" << endl;
  cout << "  5. Final Gauss fit to cleaned data - USED FOR CONTOURS" << endl;
  cout << "\nOutput files:" << endl;
  for (int w = 0; w < nWidths; ++w) {
    cout << "  - em_fits_" << widthLabels[w] << ".png" << endl;
  }
  cout << "  - em_contours_db_1sig.png  (Δβ, 1σ)" << endl;
  cout << "  - em_contours_db_3sig.png  (Δβ, 3σ)" << endl;
  cout << "  - em_contours_pb_1sig.png  (p,β, 1σ)" << endl;
  cout << "  - em_contours_pb_3sig.png  (p,β, 3σ)" << endl;
  cout << "  - em_contours_m2_1sig.png  (p,m², 1σ)" << endl;
  cout << "  - em_contours_m2_3sig.png  (p,m², 3σ)" << endl;
  cout << "  - em_contours_ma_1sig.png  (m,a, 1σ)" << endl;
  cout << "  - em_contours_ma_3sig.png  (m,a, 3σ)" << endl;
  cout << "  - em_parameters.png" << endl;
  cout << "  - em_pid_cuts_overlay.png  <-- TCutG (1σ, 3σ, 5σ) on (p, β)" << endl;
  cout << "  - em_pid_cuts_pp45_sim.root:" << endl;
  cout << "      * FitResults TTree" << endl;
  cout << "      * Contours/ (TGraphs, TSplines)" << endl;
  cout << "      * TCutG/ (1σ, 2.5σ, 3σ, 3.5σ, 5σ)" << endl;
}
