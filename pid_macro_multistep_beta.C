// pid_macro_delta_beta.C
// PID analysis using Δβ = β_measured - β_pion(p) representation
// Signal pions should peak at Δβ ≈ 0
// 
// Coordinate system:
//   X = momentum p [MeV/c]
//   Y = Δβ = β_measured - p/sqrt(p² + m_π²)
//
// Advantages:
//   - Signal centered at Δβ = 0 (intuitive)
//   - Width directly reflects velocity/timing resolution
//   - Background shape often simpler

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

// Expected β for pion at momentum p
double betaPion(double p) {
  return p / std::sqrt(p * p + gPionMass * gPionMass);
}

// Convert (p, Δβ) to actual β
double deltaBetaToBeta(double p, double dBeta) {
  return betaPion(p) + dBeta;
}

// Convert (p, β) to mass
double pBetaToMass(double p, double beta) {
  if (beta <= 0 || beta >= 1.5) return -1;
  double mass2 = p * p * (1.0 / (beta * beta) - 1.0);
  return (mass2 >= 0) ? std::sqrt(mass2) : -std::sqrt(-mass2);
}

// Convert (p, β) to (mass, a) space
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
  int phaseTag;       // 0=backward, 1=forward, 2=doubling
  std::vector<double> bkgParams;  // [bkgAmp, bkgMean, bkgSigma, poly...]
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

// =====================================================
// FITTING FUNCTION
// =====================================================

FitResult tryFitDeltaBeta(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx,
                          double pCenter, double pLow, double pHigh, double sliceWidth, int phaseTag) {
  FitResult best;
  best.success = false;
  best.mean = 0.0;
  best.sigma = 0.02;
  best.amplitude = 0.0;
  best.polyOrder = 0;
  best.chi2ndf = 1e9;
  best.entries = proj ? proj->GetEntries() : 0;
  best.pCenter = pCenter;
  best.pLow = pLow;
  best.pHigh = pHigh;
  best.sliceWidth = sliceWidth;
  best.phaseTag = phaseTag;

  if (!proj || best.entries < 30) return best;

  int bin_min = proj->FindFixBin(fitMin);
  int bin_max = proj->FindFixBin(fitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 10.0) return best;

  // =====================================================
  // Find peak position in signal region near Δβ = 0
  // Search in [-0.05, 0.05] for the pion peak
  // =====================================================
  int bin_sig_lo = proj->FindFixBin(-0.05);
  int bin_sig_hi = proj->FindFixBin(0.05);
  
  double peakVal = -1.0;
  int peakBin = 0;
  for (int b = bin_sig_lo; b <= bin_sig_hi; ++b) {
    // 3-bin smoothing
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
  int nEdgeBins = 3;
  for (int b = bin_min; b < bin_min + nEdgeBins && b <= bin_max; ++b)
    bkgLeft += proj->GetBinContent(b);
  for (int b = bin_max; b > bin_max - nEdgeBins && b >= bin_min; --b)
    bkgRight += proj->GetBinContent(b);
  bkgLeft /= nEdgeBins;
  bkgRight /= nEdgeBins;
  double bkgAvg = 0.5 * (bkgLeft + bkgRight);

  double sigAmpEst = std::max(1.0, peakVal - bkgAvg);

  // =====================================================
  // Two-pass fitting: Signal Gauss + Background Gauss + Polynomial
  // =====================================================

  struct PolyFitResult {
    bool valid;
    double chi2ndf;
    double mean, sigma, amplitude;
    double bkgGausAmp, bkgGausMean, bkgGausSigma;
    std::vector<double> polyParams;
    int polyOrder;
  };
  
  const int nModels = 3;
  PolyFitResult polyResults[nModels];
  for (int i = 0; i < nModels; ++i) polyResults[i].valid = false;

  for (int polyOrder = 0; polyOrder <= 2; ++polyOrder) {
    int nPolyParams = polyOrder + 1;
    int polyStartIdx = 6;
    
    TString funcExpr;
    switch (polyOrder) {
      case 0: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
      case 1: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x"; break;
      case 2: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x + [8]*x*x"; break;
    }

    // =====================================================
    // PASS 1: Fix signal mean to peak position
    // =====================================================
    TString funcName1 = Form("fit1_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fitFunc1 = new TF1(funcName1, funcExpr, fitMin, fitMax);
    
    // Signal Gaussian - mean fixed near peak
    fitFunc1->SetParameter(0, sigAmpEst);
    fitFunc1->SetParameter(1, peakPos);
    fitFunc1->SetParameter(2, 0.015);  // Initial sigma ~0.015
    fitFunc1->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal));
    fitFunc1->FixParameter(1, peakPos);
    fitFunc1->SetParLimits(2, 0.005, 0.08);  // σ between 0.005 and 0.08
    
    // Background Gaussian - broad, can be displaced
    fitFunc1->SetParameter(3, std::max(0.5, bkgAvg * 0.3));
    fitFunc1->SetParameter(4, 0.05);  // Start displaced
    fitFunc1->SetParameter(5, 0.08);  // Broad
    fitFunc1->SetParLimits(3, 0.0, std::max(10.0, 2.0 * peakVal));
    fitFunc1->SetParLimits(4, fitMin + 0.01, fitMax - 0.01);
    fitFunc1->SetParLimits(5, 0.04, 0.2);  // Background wider than signal
    
    // Polynomial
    fitFunc1->SetParameter(polyStartIdx, std::max(0.1, bkgAvg * 0.2));
    for (int pp = 1; pp < nPolyParams; ++pp)
      fitFunc1->SetParameter(polyStartIdx + pp, 0.0);

    TFitResultPtr result1 = proj->Fit(fitFunc1, "SQR0B");
    
    double pass1_sigAmp = fitFunc1->GetParameter(0);
    double pass1_sigSigma = fitFunc1->GetParameter(2);
    double pass1_bkgAmp = fitFunc1->GetParameter(3);
    double pass1_bkgMean = fitFunc1->GetParameter(4);
    double pass1_bkgSigma = fitFunc1->GetParameter(5);
    std::vector<double> pass1_poly;
    for (int pp = 0; pp < nPolyParams; ++pp)
      pass1_poly.push_back(fitFunc1->GetParameter(polyStartIdx + pp));

    delete fitFunc1;

    // =====================================================
    // PASS 2: Allow signal mean to vary within ±0.02 of peak
    // =====================================================
    TString funcName2 = Form("fit2_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fitFunc2 = new TF1(funcName2, funcExpr, fitMin, fitMax);
    
    fitFunc2->SetParameter(0, pass1_sigAmp);
    fitFunc2->SetParameter(1, peakPos);
    fitFunc2->SetParameter(2, pass1_sigSigma);
    fitFunc2->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal));
    // Allow mean to vary within ±0.02 of peak
    double meanLo = std::max(-0.05, peakPos - 0.02);
    double meanHi = std::min(0.05, peakPos + 0.02);
    fitFunc2->SetParLimits(1, meanLo, meanHi);
    fitFunc2->SetParLimits(2, 0.005, 0.08);
    
    fitFunc2->SetParameter(3, pass1_bkgAmp);
    fitFunc2->SetParameter(4, pass1_bkgMean);
    fitFunc2->SetParameter(5, pass1_bkgSigma);
    fitFunc2->SetParLimits(3, 0.0, std::max(10.0, 2.0 * peakVal));
    fitFunc2->SetParLimits(4, fitMin + 0.01, fitMax - 0.01);
    fitFunc2->SetParLimits(5, 0.04, 0.2);
    
    for (int pp = 0; pp < nPolyParams; ++pp)
      fitFunc2->SetParameter(polyStartIdx + pp, pass1_poly[pp]);

    TFitResultPtr result2 = proj->Fit(fitFunc2, "SQR0B");

    if (result2.Get() && result2->IsValid() && result2->Status() == 0) {
      double fMean = fitFunc2->GetParameter(1);
      double fSigma = fitFunc2->GetParameter(2);
      double fAmp = fitFunc2->GetParameter(0);
      double bkgAmp = fitFunc2->GetParameter(3);
      double bkgSigma = fitFunc2->GetParameter(5);
      int ndf = result2->Ndf();
      double chi2 = result2->Chi2();
      double chi2ndf = (ndf > 0) ? chi2 / ndf : 1e6;

      bool sigmaOK = (fSigma < bkgSigma * 0.9);
      bool meanOK = (fMean > -0.05 && fMean < 0.05);
      bool ampOK = (fAmp > 0.1);
      bool chi2OK = (chi2ndf > 0.1 && chi2ndf < 100.0);
      bool peakAligned = (std::abs(fMean - peakPos) < 0.03);
      bool ampRatioOK = (fAmp >= bkgAmp * 0.8);

      if (sigmaOK && meanOK && ampOK && chi2OK && peakAligned && ampRatioOK) {
        polyResults[polyOrder].valid = true;
        polyResults[polyOrder].chi2ndf = chi2ndf;
        polyResults[polyOrder].mean = fMean;
        polyResults[polyOrder].sigma = fSigma;
        polyResults[polyOrder].amplitude = fAmp;
        polyResults[polyOrder].bkgGausAmp = bkgAmp;
        polyResults[polyOrder].bkgGausMean = fitFunc2->GetParameter(4);
        polyResults[polyOrder].bkgGausSigma = bkgSigma;
        polyResults[polyOrder].polyOrder = polyOrder;
        polyResults[polyOrder].polyParams.clear();
        for (int pp = 0; pp < nPolyParams; ++pp)
          polyResults[polyOrder].polyParams.push_back(fitFunc2->GetParameter(polyStartIdx + pp));
      }
    }

    delete fitFunc2;
  }

  // Select best model
  const double chi2ndf_good_min = 0.5;
  const double chi2ndf_good_max = 3.0;
  
  int bestOrder = -1;
  double bestScore = 1e9;
  
  for (int po = 0; po < nModels; ++po) {
    if (polyResults[po].valid) {
      double c = polyResults[po].chi2ndf;
      if (c >= chi2ndf_good_min && c <= chi2ndf_good_max) {
        bestOrder = po;
        break;
      }
    }
  }
  
  if (bestOrder < 0) {
    for (int po = 0; po < nModels; ++po) {
      if (polyResults[po].valid) {
        double c = polyResults[po].chi2ndf;
        double score = std::abs(std::log(c)) + 0.1 * po;
        if (score < bestScore) {
          bestScore = score;
          bestOrder = po;
        }
      }
    }
  }
  
  if (bestOrder >= 0) {
    best.success = true;
    best.mean = polyResults[bestOrder].mean;
    best.sigma = polyResults[bestOrder].sigma;
    best.amplitude = polyResults[bestOrder].amplitude;
    best.polyOrder = polyResults[bestOrder].polyOrder;
    best.chi2ndf = polyResults[bestOrder].chi2ndf;
    best.bkgParams.clear();
    best.bkgParams.push_back(polyResults[bestOrder].bkgGausAmp);
    best.bkgParams.push_back(polyResults[bestOrder].bkgGausMean);
    best.bkgParams.push_back(polyResults[bestOrder].bkgGausSigma);
    for (size_t pp = 0; pp < polyResults[bestOrder].polyParams.size(); ++pp)
      best.bkgParams.push_back(polyResults[bestOrder].polyParams[pp]);
  }

  return best;
}

// =====================================================
// MAIN FUNCTION
// =====================================================

void pid_macro_multistep_beta() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain
  const char* treeName = "PimEpEm";
  const std::vector<TString> files = {
    "pp060_Sept2025.root", "pp049.root"
    //"pp060_01_exp.root","pp060_02_exp.root","pp060_03_exp.root",
    //"pp060_04_exp.root","pp060_05_exp.root","pp060_06_exp.root",
    //"pp060_07_exp.root","pp060_08_exp.root","pp060_09_exp.root", "pp060_10_exp.root"
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
  // CREATE 2D HISTOGRAM: p vs Δβ
  // Δβ = β_measured - β_pion(p)
  // β_pion(p) = p / sqrt(p² + m_π²)
  // =====================================================
  
  const char* h2name = "h2_p_deltaBeta";
  // Y-axis: Δβ from -0.15 to 0.15, 300 bins
  // X-axis: p from 0 to 1400 MeV/c, 280 bins
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

  h2DB->SetTitle("Momentum vs #Delta#beta (#beta - #beta_{#pi})");
  h2DB->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DB->GetYaxis()->SetTitle("#Delta#beta = #beta - #beta_{#pi}");

  // Also create (p, β) histogram for display
  const char* h2pb_name = "h2_p_beta";
  if (gDirectory->FindObject(h2pb_name)) gDirectory->Delete(Form("%s;*", h2pb_name));
  chain->Draw(Form("pim_beta : pim_p >> %s(280,0,1400,300,0.3,1.15)", h2pb_name), "isBest==1", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2pb_name));

  // And (mass, a) histogram
  const char* h2ma_name = "h2_mass_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) >> %s(300,0,300,160,0,16)",
    gSSquared, h2ma_name);
  if (gDirectory->FindObject(h2ma_name)) gDirectory->Delete(Form("%s;*", h2ma_name));
  chain->Draw(drawMA, "isBest==1 && pim_beta>0 && pim_beta<1", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2ma_name));

  // =====================================================
  // THREE-PHASE SCANNING PARAMETERS
  // =====================================================
  
  const double startMom = 120.0;
  const double transitionMom = 500.0;
  const double endMom = 1400.0;
  const double stepSize = 1.0;
  
  const int nWidths = 4;
  const double baseWidths[nWidths] = {5.0, 10.0, 20.0, 40.0};
  const TString widthLabels[nWidths] = {"1x(5)", "2x(10)", "4x(20)", "8x(40)"};
  const int widthColors[nWidths] = {kBlue, kRed, kGreen+2, kMagenta};

  // Fit range in Δβ
  const double fitRangeMin = -0.12;
  const double fitRangeMax = 0.12;

  // Storage
  std::vector<std::vector<FitResult>> allFitResults(nWidths);
  std::vector<std::vector<double>> allMomCenters(nWidths);

  // Canvas array
  TCanvas* cFits[nWidths];
  const int nDisplayPerWidth = 12;

  // =====================================================
  // PROCESS EACH WIDTH
  // =====================================================
  
  for (int w = 0; w < nWidths; ++w) {
    double baseW = baseWidths[w];
    cout << "\n=============================================" << endl;
    cout << "=== Width " << widthLabels[w] << " (base=" << baseW << " MeV/c) ===" << endl;
    cout << "=============================================" << endl;

    // Build slice list
    std::vector<double> momLows, momHighs, momCenters, sliceWidths;
    std::vector<int> phaseTag;

    // Phase 0: backward (right edge anchored at startMom)
    cout << "=== Phase 0: Backward from " << startMom << " MeV/c" << endl;
    {
      double pRight = startMom;
      double pLeft = startMom - baseW;
      while (pLeft >= 0) {
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(0);
        pLeft -= baseW;
      }
      // Add final slice from 0 to wherever we are
      if (momLows.back() > 0) {
        momLows.push_back(0);
        momHighs.push_back(pRight);
        momCenters.push_back(0.5 * pRight);
        sliceWidths.push_back(pRight);
        phaseTag.push_back(0);
      }
      std::reverse(momLows.begin(), momLows.end());
      std::reverse(momHighs.begin(), momHighs.end());
      std::reverse(momCenters.begin(), momCenters.end());
      std::reverse(sliceWidths.begin(), sliceWidths.end());
      std::reverse(phaseTag.begin(), phaseTag.end());
    }
    int nPhase0 = momLows.size();
    cout << "Phase 0: " << nPhase0 << " slices" << endl;

    // Phase 1: forward fixed width
    cout << "=== Phase 1: Forward " << startMom << " to " << transitionMom << " MeV/c" << endl;
    int nPhase1 = 0;
    {
      double pLeft = startMom;
      while (pLeft < transitionMom) {
        momLows.push_back(pLeft);
        momHighs.push_back(pLeft + baseW);
        momCenters.push_back(pLeft + 0.5 * baseW);
        sliceWidths.push_back(baseW);
        phaseTag.push_back(1);
        pLeft += stepSize;
        nPhase1++;
      }
    }
    cout << "Phase 1: " << nPhase1 << " slices" << endl;

    // Phase 2: doubling width
    cout << "=== Phase 2: Doubling from " << transitionMom << " MeV/c" << endl;
    int nPhase2 = 0;
    {
      double pLeft = transitionMom;
      double currentWidth = baseW;
      while (pLeft < endMom) {
        double pRight = std::min(pLeft + currentWidth, endMom);
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);
        momCenters.push_back(0.5 * (pLeft + pRight));
        sliceWidths.push_back(pRight - pLeft);
        phaseTag.push_back(2);
        cout << "  Step " << nPhase2 << ": p=[" << pLeft << ", " << pRight << "], w=" << currentWidth << endl;
        pLeft = pRight;
        currentWidth *= 2.0;
        nPhase2++;
      }
    }
    cout << "Phase 2: " << nPhase2 << " slices" << endl;

    int nSlices = momLows.size();
    cout << "Total slices: " << nSlices << endl;

    allMomCenters[w] = momCenters;
    allFitResults[w].resize(nSlices);

    // Select display indices
    std::vector<int> displayIndices;
    // From Phase 0: first, middle, last
    if (nPhase0 > 0) {
      displayIndices.push_back(0);
      displayIndices.push_back(nPhase0 / 2);
      displayIndices.push_back(nPhase0 - 1);
    }
    // From Phase 1: evenly spaced
    for (int k = 0; k < 5 && k * nPhase1 / 5 < nPhase1; ++k) {
      int idx = nPhase0 + k * nPhase1 / 5;
      displayIndices.push_back(idx);
    }
    // From Phase 2: all
    for (int k = 0; k < nPhase2 && k < 4; ++k) {
      displayIndices.push_back(nPhase0 + nPhase1 + k);
    }
    // Remove duplicates and sort
    std::sort(displayIndices.begin(), displayIndices.end());
    displayIndices.erase(std::unique(displayIndices.begin(), displayIndices.end()), displayIndices.end());
    if (displayIndices.size() > (size_t)nDisplayPerWidth)
      displayIndices.resize(nDisplayPerWidth);

    // Create canvas
    cFits[w] = new TCanvas(Form("c_fits_dBeta_%d", w), 
                           Form("Delta-Beta Fits - %s", widthLabels[w].Data()), 1400, 900);
    cFits[w]->Divide(4, 3);

    // Fit all slices
    int nSuccess = 0;
    for (int i = 0; i < nSlices; ++i) {
      int xbin_lo = h2DB->GetXaxis()->FindFixBin(momLows[i] + 0.01);
      int xbin_hi = h2DB->GetXaxis()->FindFixBin(momHighs[i] - 0.01);
      
      TString projName = Form("proj_db_w%d_s%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DB->ProjectionY(projName, xbin_lo, xbin_hi, "e");

      allFitResults[w][i] = tryFitDeltaBeta(proj, fitRangeMin, fitRangeMax, i, w,
                                             momCenters[i], momLows[i], momHighs[i], 
                                             sliceWidths[i], phaseTag[i]);
      
      if (allFitResults[w][i].success) nSuccess++;

      // Print progress for selected slices
      TString phaseStr = (phaseTag[i] == 0) ? "[P0]" : ((phaseTag[i] == 1) ? "[P1]" : "[P2]");
      if (i < 3 || i == nPhase0 - 1 || i == nPhase0 || i >= nSlices - 3 || phaseTag[i] == 2) {
        if (allFitResults[w][i].success) {
          cout << Form("  Slice %4d: p=[%6.0f,%6.0f], w=%5.0f, μ=%7.4f, σ=%6.4f, χ²/n=%.2f %s", 
                       i, momLows[i], momHighs[i], sliceWidths[i],
                       allFitResults[w][i].mean, allFitResults[w][i].sigma,
                       allFitResults[w][i].chi2ndf, phaseStr.Data()) << endl;
        } else {
          cout << Form("  Slice %4d: p=[%6.0f,%6.0f], FAIL %s", 
                       i, momLows[i], momHighs[i], phaseStr.Data()) << endl;
        }
      } else if (i == 3) {
        cout << "  ... (more slices) ..." << endl;
      }
      delete proj;
    }
    
    cout << "Success: " << nSuccess << "/" << nSlices << endl;

    // Display fits
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

      // Zero line (Δβ = 0 is the pion position)
      TLine* zeroLine = new TLine(0, 0, 0, proj->GetMaximum() * 1.1);
      zeroLine->SetLineColor(kGray+1);
      zeroLine->SetLineStyle(kDashed);
      zeroLine->Draw("same");

      if (result.success && result.bkgParams.size() >= 3) {
        TString funcExpr;
        int nPolyParams = result.polyOrder + 1;
        switch (result.polyOrder) {
          case 0: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
          case 1: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x"; break;
          case 2: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x + [8]*x*x"; break;
          default: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
        }
        
        TF1* fitFunc = new TF1(Form("fitdisp_db_w%d_d%d", w, (int)d), funcExpr, fitRangeMin, fitRangeMax);
        fitFunc->SetParameter(0, result.amplitude);
        fitFunc->SetParameter(1, result.mean);
        fitFunc->SetParameter(2, result.sigma);
        fitFunc->SetParameter(3, result.bkgParams[0]);
        fitFunc->SetParameter(4, result.bkgParams[1]);
        fitFunc->SetParameter(5, result.bkgParams[2]);
        for (int pp = 0; pp < nPolyParams; ++pp)
          fitFunc->SetParameter(6 + pp, result.bkgParams[3 + pp]);
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
        TF1* bkgGaus = new TF1(Form("bkg_db_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
        bkgGaus->SetParameters(result.bkgParams[0], result.bkgParams[1], result.bkgParams[2]);
        bkgGaus->SetLineColor(kOrange+1);
        bkgGaus->SetLineStyle(kDashed);
        bkgGaus->SetLineWidth(2);
        bkgGaus->Draw("same");

        // Polynomial (green dashed)
        TString polyExpr;
        switch (result.polyOrder) {
          case 0: polyExpr = "[0]"; break;
          case 1: polyExpr = "[0] + [1]*x"; break;
          case 2: polyExpr = "[0] + [1]*x + [2]*x*x"; break;
          default: polyExpr = "[0]"; break;
        }
        TF1* poly = new TF1(Form("poly_db_w%d_d%d", w, (int)d), polyExpr, fitRangeMin, fitRangeMax);
        for (int pp = 0; pp < nPolyParams; ++pp)
          poly->SetParameter(pp, result.bkgParams[3 + pp]);
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
        tex.SetTextColor(kBlack);
        tex.DrawLatex(0.55, 0.65, Form("pol%d", result.polyOrder));
        if (result.chi2ndf >= 0.5 && result.chi2ndf <= 3.0) {
          tex.SetTextColor(kGreen+2);
        } else if (result.chi2ndf > 3.0 && result.chi2ndf <= 10.0) {
          tex.SetTextColor(kOrange+1);
        } else {
          tex.SetTextColor(kRed);
        }
        tex.DrawLatex(0.55, 0.58, Form("#chi^{2}/n=%.1f", result.chi2ndf));
        tex.SetTextColor(kBlack);
      } else {
        tex.SetTextColor(kRed);
        tex.DrawLatex(0.55, 0.77, "Fit failed");
        tex.SetTextColor(kBlack);
      }
      tex.DrawLatex(0.55, 0.51, Form("N=%.0f", result.entries));
      
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
  legDB_1->AddEntry(zeroLineDB1, "#Delta#beta=0 (pion)", "l");
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
  // β = β_pion(p) + Δβ
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
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, betaPointsUpper;
    std::vector<double> pPointsLower, betaPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      // Upper: mean + σ (in Δβ) → β_pion + mean + σ
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      if (beta_upper > 0.3 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      // Lower: mean - σ (in Δβ) → β_pion + mean - σ
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
    
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> massPointsUpper, aPointsUpper;
    std::vector<double> massPointsLower, aPointsLower;
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double beta_pion = betaPion(p);
      
      // Upper boundary
      double beta_upper = beta_pion + smoothMean[i] + 1.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > 0 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      // Lower boundary
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
  
  TCanvas* cPar = new TCanvas("c_par", "Fit Parameters vs Momentum", 1200, 450);
  cPar->Divide(3, 1);

  // Mean (should be ~0)
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

  // Chi2/ndf
  cPar->cd(3);
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
  cCombMA_1sig->SaveAs("pion_massa_combined_1sigma.png");
  cCombMA_3sig->SaveAs("pion_massa_combined_3sigma.png");
  cPar->SaveAs("pion_parameters_dBeta.png");

  // Output file
  std::ofstream outfile("pion_fit_results_dBeta.txt");
  outfile << "Width\tpCenter\tpLow\tpHigh\tPhase\tMean_dBeta\tSigma_dBeta\tChi2NDF\tStatus" << std::endl;
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allMomCenters[w].size(); ++i) {
      FitResult& r = allFitResults[w][i];
      TString phaseStr = (r.phaseTag == 0) ? "BWD" : ((r.phaseTag == 1) ? "FWD" : "DBL");
      outfile << widthLabels[w] << "\t" << r.pCenter << "\t" << r.pLow << "\t" << r.pHigh << "\t"
              << phaseStr << "\t" << r.mean << "\t" << r.sigma << "\t" << r.chi2ndf << "\t"
              << (r.success ? "OK" : "FAIL") << std::endl;
    }
  }
  outfile.close();

  cout << "\n=== Analysis Complete ===" << endl;
  cout << "FIT MODEL: Δβ = β_measured - β_pion(p)" << endl;
  cout << "  - Signal Gaussian: centered near Δβ=0, narrow σ" << endl;
  cout << "  - Background Gaussian: broad, can be displaced" << endl;
  cout << "  - Polynomial: pol0-pol2" << endl;
  cout << "\nOutput files:" << endl;
  cout << "  - pion_fits_dBeta_*.png" << endl;
  cout << "  - pion_dBeta_combined_*.png" << endl;
  cout << "  - pion_pbeta_combined_*.png" << endl;
  cout << "  - pion_massa_combined_*.png" << endl;
  cout << "  - pion_parameters_dBeta.png" << endl;
  cout << "  - pion_fit_results_dBeta.txt" << endl;
}
