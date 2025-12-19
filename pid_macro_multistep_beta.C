// pid_macro_momentum_slice.C
// ADAPTIVE APPROACH:
// - Phase 1 (p < 500 MeV/c): Regular sliding scan with fixed width
// - Phase 2 (p >= 500 MeV/c): Width doubles with each step
// This handles decreasing statistics at high momentum
//
// Widths: 1x=5 MeV/c, 2x=10 MeV/c, 4x=20 MeV/c, 8x=40 MeV/c (base widths)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

// Physical constants
const double gPionMass = 139.57;  // MeV/c²
const double gPionMass2 = gPionMass * gPionMass;
const double gSSquared = 1e-4;

// Transition momentum
const double gTransitionMom = 500.0;  // MeV/c

// Structure to hold fit results
struct FitResult {
  bool success;
  double mean;
  double sigma;
  double amplitude;
  int polyOrder;
  double chi2ndf;
  std::vector<double> bkgParams;
  double entries;
  double pCenter;
  double pLow;
  double pHigh;
  double sliceWidth;
  int phaseTag;  // 0 = backward, 1 = forward fixed, 2 = doubling
};

// Convert (p, Δm²) → β
bool deltaM2ToBeta(double p, double deltaM2, double& beta_out) {
  double m2 = deltaM2 + gPionMass2;
  if (m2 < 0) {
    beta_out = p / std::sqrt(p * p + m2);
    return (beta_out > 0 && beta_out < 2.0);
  }
  double E2 = p * p + m2;
  if (E2 <= 0) return false;
  beta_out = p / std::sqrt(E2);
  return (beta_out > 0 && beta_out < 2.0);
}

// Convert (p, β) → (mass, a)
bool pBetaToMassA(double p, double beta, double s2, double& mass_out, double& a_out) {
  if (beta <= 0 || beta >= 2.0) return false;
  
  double beta2 = beta * beta;
  double factor = 1.0 / beta2 - 1.0;
  
  if (factor < 0) {
    mass_out = -p * std::sqrt(-factor);
  } else {
    mass_out = p * std::sqrt(factor);
  }
  
  double a2 = 1.0 + s2 * p * p - (1.0 - beta2) * (1.0 - beta2);
  if (a2 < 0) a2 = 0;
  a_out = std::sqrt(a2);
  
  return true;
}

// Fit with polynomial background selection
// Strategy: 
// 1. Find peak position PRECISELY in signal region [-8000, 8000]
// 2. PASS 1: Fix signal mean to peak, fit all other parameters
// 3. PASS 2: Allow signal mean to vary within narrow range (±2000) around peak
// 4. Background Gauss is free to be displaced anywhere
// 5. Use extended fit range for low momentum (p < 300 MeV/c)

FitResult tryFitWithPolynomials(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx,
                                 double pCenter, double pLow, double pHigh, double sliceWidth, int phaseTag) {
  FitResult best;
  best.success = false;
  best.mean = 0.0;
  best.sigma = 5000.0;
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

  // Adjust fit range for low momentum (more asymmetric background)
  double localFitMin = fitMin;
  double localFitMax = fitMax;
  if (pCenter < 300.0) {
    localFitMin = -15000.0;
    localFitMax = 30000.0;
  }

  int bin_min = proj->FindFixBin(localFitMin);
  int bin_max = proj->FindFixBin(localFitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 10.0) return best;

  // =====================================================
  // STEP 1: Find peak position PRECISELY in signal region [-8000, 8000]
  // Use 3-bin smoothing to be robust against statistical fluctuations
  // =====================================================
  int bin_sig_lo = proj->FindFixBin(-8000.0);
  int bin_sig_hi = proj->FindFixBin(8000.0);
  
  double peakVal_signal = -1.0;
  int peakBin_signal = 0;
  for (int b = bin_sig_lo + 1; b <= bin_sig_hi - 1; ++b) {
    double v = (proj->GetBinContent(b-1) + proj->GetBinContent(b) + proj->GetBinContent(b+1)) / 3.0;
    if (v > peakVal_signal) { 
      peakVal_signal = v; 
      peakBin_signal = b; 
    }
  }
  double peakPos_signal = (peakBin_signal > 0) ? proj->GetBinCenter(peakBin_signal) : 0.0;
  
  // Get actual (non-smoothed) peak value for amplitude estimate
  peakVal_signal = (peakBin_signal > 0) ? proj->GetBinContent(peakBin_signal) : 1.0;

  // Estimate background from edges
  double bkgLeft = 0, bkgRight = 0;
  int nEdgeBins = 5;
  for (int b = bin_min; b < bin_min + nEdgeBins && b <= bin_max; ++b)
    bkgLeft += proj->GetBinContent(b);
  for (int b = bin_max; b > bin_max - nEdgeBins && b >= bin_min; --b)
    bkgRight += proj->GetBinContent(b);
  bkgLeft /= nEdgeBins;
  bkgRight /= nEdgeBins;
  double bkgAvg = 0.5 * (bkgLeft + bkgRight);

  // Estimate signal amplitude
  double sigAmpEst = std::max(1.0, peakVal_signal - bkgAvg);

  // =====================================================
  // STEP 2: Two-pass fitting for each polynomial order
  // =====================================================

  struct PolyFitResult {
    bool valid;
    double chi2ndf;
    double mean, sigma, amplitude;
    double bkgGausAmp, bkgGausMean, bkgGausSigma;
    std::vector<double> polyParams;
    int ndf;
    int polyOrder;
  };
  
  const int nModels = 3;  // pol0, pol1, pol2
  PolyFitResult polyResults[nModels];
  for (int i = 0; i < nModels; ++i) polyResults[i].valid = false;

  for (int polyOrder = 0; polyOrder <= 2; ++polyOrder) {
    int nPolyParams = polyOrder + 1;
    int polyStartIdx = 6;
    
    // Build function: gaus(0) + gaus(3) + polN(6)
    TString funcExpr;
    switch (polyOrder) {
      case 0: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
      case 1: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x"; break;
      case 2: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x + [8]*x*x"; break;
    }

    // =====================================================
    // PASS 1: Fix signal mean to peak position, fit everything else
    // =====================================================
    TString funcName1 = Form("tryfit1_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fitFunc1 = new TF1(funcName1, funcExpr, localFitMin, localFitMax);
    
    // Signal Gaussian - FIX mean to peak
    fitFunc1->SetParameter(0, sigAmpEst);
    fitFunc1->SetParameter(1, peakPos_signal);
    fitFunc1->SetParameter(2, 4000.0);
    fitFunc1->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal_signal));
    fitFunc1->FixParameter(1, peakPos_signal);  // *** FIX SIGNAL MEAN ***
    fitFunc1->SetParLimits(2, 1500.0, 10000.0);
    
    // Background Gaussian - free to move anywhere
    fitFunc1->SetParameter(3, std::max(0.5, bkgAvg * 0.5));
    fitFunc1->SetParameter(4, 5000.0);  // Start away from signal
    fitFunc1->SetParameter(5, 20000.0);
    fitFunc1->SetParLimits(3, 0.0, std::max(10.0, 2.0 * peakVal_signal));
    fitFunc1->SetParLimits(4, localFitMin + 1000, localFitMax - 1000);
    fitFunc1->SetParLimits(5, 12000.0, 45000.0);
    
    // Polynomial
    fitFunc1->SetParameter(polyStartIdx, std::max(0.1, bkgAvg * 0.2));
    for (int pp = 1; pp < nPolyParams; ++pp)
      fitFunc1->SetParameter(polyStartIdx + pp, 0.0);

    TFitResultPtr result1 = proj->Fit(fitFunc1, "SQR0B");
    
    // Store pass 1 results
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
    // PASS 2: Allow signal mean to vary within ±2000 of peak
    // Use pass 1 results as starting point
    // =====================================================
    TString funcName2 = Form("tryfit2_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TF1* fitFunc2 = new TF1(funcName2, funcExpr, localFitMin, localFitMax);
    
    // Signal Gaussian - allow SMALL variation around peak
    fitFunc2->SetParameter(0, pass1_sigAmp);
    fitFunc2->SetParameter(1, peakPos_signal);
    fitFunc2->SetParameter(2, pass1_sigSigma);
    fitFunc2->SetParLimits(0, 0.1, std::max(10.0, 5.0 * peakVal_signal));
    // Constrain mean to ±2000 of peak position
    double meanLo = std::max(-8000.0, peakPos_signal - 2000.0);
    double meanHi = std::min(8000.0, peakPos_signal + 2000.0);
    fitFunc2->SetParLimits(1, meanLo, meanHi);
    fitFunc2->SetParLimits(2, 1500.0, 10000.0);
    
    // Background Gaussian - use pass 1 as starting point
    fitFunc2->SetParameter(3, pass1_bkgAmp);
    fitFunc2->SetParameter(4, pass1_bkgMean);
    fitFunc2->SetParameter(5, pass1_bkgSigma);
    fitFunc2->SetParLimits(3, 0.0, std::max(10.0, 2.0 * peakVal_signal));
    fitFunc2->SetParLimits(4, localFitMin + 1000, localFitMax - 1000);
    fitFunc2->SetParLimits(5, 12000.0, 45000.0);
    
    // Polynomial - use pass 1 as starting point
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

      // Validation
      bool sigmaOK = (fSigma < bkgSigma * 0.9);
      bool meanOK = (fMean > -8000.0 && fMean < 8000.0);
      bool ampOK = (fAmp > 0.1);
      bool chi2OK = (chi2ndf > 0.1 && chi2ndf < 100.0);
      // Verify signal mean stayed close to data peak
      bool peakAligned = (std::abs(fMean - peakPos_signal) < 3000.0);
      // CRITICAL: Signal amplitude must be >= background amplitude
      // (otherwise "background" is actually the signal!)
      bool ampRatioOK = (fAmp >= bkgAmp * 0.8);  // Signal should dominate

      if (sigmaOK && meanOK && ampOK && chi2OK && peakAligned && ampRatioOK) {
        polyResults[polyOrder].valid = true;
        polyResults[polyOrder].chi2ndf = chi2ndf;
        polyResults[polyOrder].mean = fMean;
        polyResults[polyOrder].sigma = fSigma;
        polyResults[polyOrder].amplitude = fAmp;
        polyResults[polyOrder].bkgGausAmp = fitFunc2->GetParameter(3);
        polyResults[polyOrder].bkgGausMean = fitFunc2->GetParameter(4);
        polyResults[polyOrder].bkgGausSigma = bkgSigma;
        polyResults[polyOrder].ndf = ndf;
        polyResults[polyOrder].polyOrder = polyOrder;
        polyResults[polyOrder].polyParams.clear();
        for (int pp = 0; pp < nPolyParams; ++pp)
          polyResults[polyOrder].polyParams.push_back(fitFunc2->GetParameter(polyStartIdx + pp));
      }
    }

    delete fitFunc2;
  }

  // =====================================================
  // STEP 3: Select best model (prefer simplest with good chi2/ndf)
  // =====================================================
  
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

TString getBkgFuncExpr(int polyOrder) {
  switch (polyOrder) {
    case 0: return "[0]";
    case 1: return "pol1";
    case 2: return "pol2";
    case 3: return "pol3";
    case 4: return "pol4";
    default: return "pol1";
  }
}

// =====================================================
// Smoothing functions for contour boundaries
// =====================================================

// Running median filter (robust to outliers)
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

// Gaussian-weighted smoothing
std::vector<double> gaussianSmooth(const std::vector<double>& data, int windowSize, double sigma) {
  std::vector<double> result(data.size());
  int halfWin = windowSize / 2;
  
  // Build Gaussian kernel
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

// Combined: first median to remove outliers, then Gaussian to smooth
std::vector<double> robustSmooth(const std::vector<double>& data, int medianWin, int gaussWin) {
  std::vector<double> temp = runningMedian(data, medianWin);
  return gaussianSmooth(temp, gaussWin, gaussWin / 3.0);
}

void pid_macro_multistep_beta() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain
  const char* treeName = "PimEpEm_ID";
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
    cout << "No entries in TChain." << endl;
    return;
  }
  cout << "TChain: files=" << added << ", entries=" << nEnt << endl;

  // --- Create 2D histogram: X = momentum, Y = Δm²
  const char* h2DM2name = "h2_mom_vs_deltam2";
  TString drawDM2 = Form(
    "pim_p*pim_p*(1.0/(pim_beta*pim_beta) - 1.0) - %.2f : pim_p >> %s(700,0,1400,500,-30000,50000)",
    gPionMass2, h2DM2name);

  if (gDirectory->FindObject(h2DM2name)) gDirectory->Delete(Form("%s;*", h2DM2name));
  chain->Draw(drawDM2, "isBest==1", "colz");
  TH2F* h2DM2 = static_cast<TH2F*>(gDirectory->Get(h2DM2name));
  if (!h2DM2) { cout << "Failed to create Δm² histogram" << endl; return; }

  h2DM2->SetTitle("#Delta m^{2} vs Momentum (pion line at #Delta m^{2} = 0)");
  h2DM2->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DM2->GetYaxis()->SetTitle("#Delta m^{2} = p^{2}(1/#beta^{2}-1) - m_{#pi}^{2} [MeV^{2}/c^{4}]");

  // --- Also create (p, β) histogram
  const char* h2PBname = "h2_mom_vs_beta";
  if (gDirectory->FindObject(h2PBname)) gDirectory->Delete(Form("%s;*", h2PBname));
  chain->Draw(Form("pim_beta : pim_p >> %s(700,0,1400,500,0.4,1.15)", h2PBname), "isBest==1", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2PBname));
  if (h2PB) {
    h2PB->SetTitle("Momentum vs #beta");
    h2PB->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2PB->GetYaxis()->SetTitle("#beta");
  }

  // --- Create (mass, a) histogram
  const char* h2MAname = "h2_mass_vs_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) * sign(pim_p) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) * sign(pim_p) >> %s(300,0,300,700,0,16)",
    gSSquared, h2MAname);
  if (gDirectory->FindObject(h2MAname)) gDirectory->Delete(Form("%s;*", h2MAname));
  chain->Draw(drawMA, "isBest==1", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2MAname));
  if (h2MA) {
    h2MA->SetTitle(Form("Mass vs a-parameter (s^{2} = %.1e)", gSSquared));
    h2MA->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
    h2MA->GetYaxis()->SetTitle("a * sign(p)");
  }

  // --- Parameters
  const double baseWidth = 5.0;  // MeV/c for 1x
  const double stepSize = 1.0;   // MeV/c step for phase 1
  const double startMom = 120.0; // Start momentum (left edge)
  const double transitionMom = gTransitionMom;  // 500 MeV/c
  const double endMom = 1400.0;  // Maximum momentum
  
  const double fitRangeMin = -20000.0;
  const double fitRangeMax = 30000.0;

  // Width multipliers
  const int nWidths = 4;
  double widthMultipliers[nWidths] = {1.0, 2.0, 4.0, 8.0};
  TString widthLabels[nWidths] = {"1x(5)", "2x(10)", "4x(20)", "8x(40)"};
  int widthColors[nWidths] = {kBlue, kGreen+2, kOrange+1, kRed};

  // --- Storage
  std::vector<std::vector<double>> allMomCenters(nWidths);
  std::vector<std::vector<FitResult>> allFitResults(nWidths);

  // --- Create fit canvases
  TCanvas* cFits[nWidths];
  const int nDisplayPerWidth = 12;
  
  for (int w = 0; w < nWidths; ++w) {
    cFits[w] = new TCanvas(Form("c_fit_%d", w),
                           Form("Pion Fits - %s (P0:backward, P1:fixed, P2:doubling)", 
                                widthLabels[w].Data()),
                           1200, 800);
    cFits[w]->Divide(4, 3);
  }

  // --- Perform scans with adaptive widths
  for (int w = 0; w < nWidths; ++w) {
    double baseW = baseWidth * widthMultipliers[w];
    
    cout << "\n=============================================" << endl;
    cout << "=== Width " << widthLabels[w] << " (base=" << baseW << " MeV/c) ===" << endl;
    cout << "=== Phase 1: p < " << transitionMom << " MeV/c, fixed width" << endl;
    cout << "=== Phase 2: p >= " << transitionMom << " MeV/c, width doubles each step" << endl;
    cout << "=============================================" << endl;

    // Build slice list
    std::vector<double> momCenters;
    std::vector<double> momLows;
    std::vector<double> momHighs;
    std::vector<double> sliceWidths;
    std::vector<int> phaseTag;  // 0 = backward, 1 = forward fixed, 2 = doubling
    
    // Phase 0: Backward fitting from startMom (120) towards 0
    // Right edge anchored at startMom, left edge extends progressively
    // For 1x(5): [115-120], [110-120], [105-120], ..., [0-120]
    // For 2x(10): [110-120], [100-120], [90-120], ..., [0-120]
    // etc.
    
    std::vector<double> phase0Lows, phase0Highs, phase0Centers, phase0Widths;
    
    double pRight0 = startMom;  // Anchored at 120
    double pLeft0 = startMom - baseW;  // First slice: [115-120] for 1x
    
    while (pLeft0 >= 0) {
      double currentWidth0 = pRight0 - pLeft0;
      double pCenter0 = 0.5 * (pLeft0 + pRight0);
      
      phase0Lows.push_back(pLeft0);
      phase0Highs.push_back(pRight0);
      phase0Centers.push_back(pCenter0);
      phase0Widths.push_back(currentWidth0);
      
      // Extend left edge by baseW
      pLeft0 -= baseW;
    }
    
    // Add phase 0 in reverse order (so we go from low momentum to high)
    for (int i = (int)phase0Lows.size() - 1; i >= 0; --i) {
      momLows.push_back(phase0Lows[i]);
      momHighs.push_back(phase0Highs[i]);
      momCenters.push_back(phase0Centers[i]);
      sliceWidths.push_back(phase0Widths[i]);
      phaseTag.push_back(0);  // Phase 0
    }
    
    int nPhase0 = momCenters.size();
    cout << "Phase 0 (backward): " << nPhase0 << " slices, right edge anchored at " << startMom << endl;
    if (nPhase0 > 0) {
      cout << "  First slice: [" << momLows[0] << ", " << momHighs[0] << "], width=" << sliceWidths[0] << endl;
      cout << "  Last slice:  [" << momLows[nPhase0-1] << ", " << momHighs[nPhase0-1] << "], width=" << sliceWidths[nPhase0-1] << endl;
    }
    
    // Phase 1: Regular sliding scan from startMom until left edge reaches transitionMom
    double pLeft = startMom;
    while (pLeft < transitionMom) {
      double pRight = pLeft + baseW;
      double pCenter = 0.5 * (pLeft + pRight);
      
      momLows.push_back(pLeft);
      momHighs.push_back(pRight);
      momCenters.push_back(pCenter);
      sliceWidths.push_back(baseW);
      phaseTag.push_back(1);  // Phase 1
      
      pLeft += stepSize;
    }
    
    int nPhase1 = momCenters.size() - nPhase0;
    cout << "Phase 1 (forward fixed): " << nPhase1 << " slices (p_left from " << startMom << " to " << transitionMom << ")" << endl;
    
    // Phase 2: Doubling width with each step
    pLeft = transitionMom;
    double currentWidth = baseW;
    int doublingCount = 0;
    
    while (pLeft < endMom && currentWidth < 1000) {  // Safety limit on width
      double pRight = pLeft + currentWidth;
      if (pRight > endMom + 100) break;  // Don't go too far
      
      double pCenter = 0.5 * (pLeft + pRight);
      
      momLows.push_back(pLeft);
      momHighs.push_back(pRight);
      momCenters.push_back(pCenter);
      sliceWidths.push_back(currentWidth);
      phaseTag.push_back(2);  // Phase 2
      
      cout << Form("  Doubling step %d: p=[%.0f, %.0f], width=%.0f", 
                   doublingCount, pLeft, pRight, currentWidth) << endl;
      
      // Move to next slice and double width
      pLeft = pRight;
      currentWidth *= 2.0;
      doublingCount++;
    }
    
    int nSlices = momCenters.size();
    int nPhase2 = nSlices - nPhase0 - nPhase1;
    cout << "Phase 2 (doubling): " << nPhase2 << " slices" << endl;
    cout << "Total slices: " << nSlices << endl;

    allMomCenters[w] = momCenters;
    allFitResults[w].resize(nSlices);

    // Display indices - show samples from each phase
    std::vector<int> displayIndices;
    
    // Phase 0: show 2-3 samples
    int nDisplayP0 = std::min(3, nPhase0);
    for (int d = 0; d < nDisplayP0; ++d) {
      int idx = (nPhase0 > 1) ? (int)((double)d * (nPhase0 - 1) / (nDisplayP0 - 1) + 0.5) : 0;
      displayIndices.push_back(idx);
    }
    
    // Phase 1: show 4-5 samples
    int nDisplayP1 = std::min(5, nPhase1);
    for (int d = 0; d < nDisplayP1; ++d) {
      int idx = nPhase0 + ((nPhase1 > 1) ? (int)((double)d * (nPhase1 - 1) / (nDisplayP1 - 1) + 0.5) : 0);
      displayIndices.push_back(idx);
    }
    
    // Phase 2: show all (typically 6-8)
    for (int i = nPhase0 + nPhase1; i < nSlices && (int)displayIndices.size() < nDisplayPerWidth; ++i) {
      displayIndices.push_back(i);
    }
    
    cout << "Display indices: ";
    for (int idx : displayIndices) {
      TString phaseStr = (phaseTag[idx] == 0) ? "P0" : ((phaseTag[idx] == 1) ? "P1" : "P2");
      cout << idx << "(" << phaseStr << ",p=" << momCenters[idx] << ",w=" << sliceWidths[idx] << ") ";
    }
    cout << endl;

    // Perform all fits
    int nSuccess = 0, nSuccessP0 = 0, nSuccessP1 = 0, nSuccessP2 = 0;
    for (int i = 0; i < nSlices; ++i) {
      int xbin_lo = h2DM2->GetXaxis()->FindFixBin(momLows[i] + 0.01);
      int xbin_hi = h2DM2->GetXaxis()->FindFixBin(momHighs[i] - 0.01);
      
      TString projName = Form("proj_fit_w%d_%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DM2->ProjectionY(projName, xbin_lo, xbin_hi, "e");

      allFitResults[w][i] = tryFitWithPolynomials(proj, fitRangeMin, fitRangeMax, i, w,
                                                   momCenters[i], momLows[i], momHighs[i], 
                                                   sliceWidths[i], phaseTag[i]);
      
      if (allFitResults[w][i].success) {
        nSuccess++;
        if (phaseTag[i] == 0) nSuccessP0++;
        else if (phaseTag[i] == 1) nSuccessP1++;
        else nSuccessP2++;
      }

      // Print progress
      TString phaseStr = (phaseTag[i] == 0) ? "[P0]" : ((phaseTag[i] == 1) ? "[P1]" : "[P2]");
      if (i < 3 || i == nPhase0 - 1 || i == nPhase0 || i == nPhase0 + nPhase1 - 1 || 
          i == nPhase0 + nPhase1 || i >= nSlices - 3 || phaseTag[i] == 2) {
        if (allFitResults[w][i].success && allFitResults[w][i].bkgParams.size() >= 3) {
          cout << Form("  Slice %4d: p=[%6.0f,%6.0f], w=%5.0f, N=%6.0f, OK(pol%d,μ=%5.0f,σ_sig=%5.0f,σ_bkg=%5.0f,χ²/n=%.2f) %s", 
                       i, momLows[i], momHighs[i], sliceWidths[i], allFitResults[w][i].entries,
                       allFitResults[w][i].polyOrder, allFitResults[w][i].mean, 
                       allFitResults[w][i].sigma, allFitResults[w][i].bkgParams[2],
                       allFitResults[w][i].chi2ndf, phaseStr.Data()) << endl;
        } else {
          cout << Form("  Slice %4d: p=[%6.0f,%6.0f], w=%5.0f, N=%6.0f, FAIL %s", 
                       i, momLows[i], momHighs[i], sliceWidths[i], allFitResults[w][i].entries,
                       phaseStr.Data()) << endl;
        }
      } else if (i == 3) {
        cout << "  ... (more slices) ..." << endl;
      }
      delete proj;
    }
    
    cout << "Success: " << nSuccess << "/" << nSlices 
         << " (P0: " << nSuccessP0 << "/" << nPhase0 
         << ", P1: " << nSuccessP1 << "/" << nPhase1 
         << ", P2: " << nSuccessP2 << "/" << nPhase2 << ")" << endl;

    // Display fits
    for (size_t d = 0; d < displayIndices.size() && d < (size_t)nDisplayPerWidth; ++d) {
      int i = displayIndices[d];
      FitResult& result = allFitResults[w][i];

      cFits[w]->cd(d + 1);
      gPad->SetLeftMargin(0.14);
      gPad->SetRightMargin(0.04);
      gPad->SetTopMargin(0.10);
      gPad->SetBottomMargin(0.12);

      // Determine local fit range (extended for low momentum)
      double localFitMin = fitRangeMin;
      double localFitMax = fitRangeMax;
      if (result.pCenter < 300.0) {
        localFitMin = -15000.0;
        localFitMax = 30000.0;
      }

      int xbin_lo = h2DM2->GetXaxis()->FindFixBin(result.pLow + 0.01);
      int xbin_hi = h2DM2->GetXaxis()->FindFixBin(result.pHigh - 0.01);
      
      TString projName = Form("proj_disp_w%d_d%d", w, (int)d);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2DM2->ProjectionY(projName, xbin_lo, xbin_hi, "e");
      if (!proj) continue;

      TString phaseStr = (result.phaseTag == 0) ? " [BWD]" : ((result.phaseTag == 1) ? "" : " [DBL]");
      proj->SetTitle(Form("p=[%.0f,%.0f] w=%.0f%s", result.pLow, result.pHigh, result.sliceWidth, phaseStr.Data()));
      proj->GetXaxis()->SetRangeUser(localFitMin, localFitMax);
      proj->GetXaxis()->SetTitle("#Delta m^{2} [MeV^{2}/c^{4}]");
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

      if (result.success && result.bkgParams.size() >= 3) {
        // Build function expression: gaus(0) + gaus(3) + polN(6)
        TString funcExpr;
        int nPolyParams = result.polyOrder + 1;
        switch (result.polyOrder) {
          case 0: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
          case 1: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x"; break;
          case 2: funcExpr = "gaus(0) + gaus(3) + [6] + [7]*x + [8]*x*x"; break;
          default: funcExpr = "gaus(0) + gaus(3) + [6]"; break;
        }
        
        TF1* fitFunc = new TF1(Form("fitdisp_w%d_d%d", w, (int)d), funcExpr, localFitMin, localFitMax);
        // Signal Gaussian (params 0,1,2)
        fitFunc->SetParameter(0, result.amplitude);
        fitFunc->SetParameter(1, result.mean);
        fitFunc->SetParameter(2, result.sigma);
        // Background Gaussian (params 3,4,5) - stored in bkgParams[0,1,2]
        fitFunc->SetParameter(3, result.bkgParams[0]);  // bkg amp
        fitFunc->SetParameter(4, result.bkgParams[1]);  // bkg mean
        fitFunc->SetParameter(5, result.bkgParams[2]);  // bkg sigma
        // Polynomial (params 6+) - stored in bkgParams[3+]
        for (int pp = 0; pp < nPolyParams; ++pp)
          fitFunc->SetParameter(6 + pp, result.bkgParams[3 + pp]);
        fitFunc->SetLineColor(kRed);
        fitFunc->SetLineWidth(2);
        fitFunc->Draw("same");

        // Signal Gaussian only (blue solid, thicker for visibility)
        TF1* sig = new TF1(Form("sig_w%d_d%d", w, (int)d), "gaus", localFitMin, localFitMax);
        sig->SetParameters(result.amplitude, result.mean, result.sigma);
        sig->SetLineColor(kBlue);
        sig->SetLineStyle(1);  // Solid line for signal
        sig->SetLineWidth(3);  // Thicker
        sig->Draw("same");

        // Background Gaussian only (cyan dashed)
        TF1* bkgGaus = new TF1(Form("bkggaus_w%d_d%d", w, (int)d), "gaus", localFitMin, localFitMax);
        bkgGaus->SetParameters(result.bkgParams[0], result.bkgParams[1], result.bkgParams[2]);
        bkgGaus->SetLineColor(kOrange+1);
        bkgGaus->SetLineStyle(kDashed);
        bkgGaus->SetLineWidth(2);
        bkgGaus->Draw("same");

        // Polynomial only (green dashed)
        TString polyExpr;
        switch (result.polyOrder) {
          case 0: polyExpr = "[0]"; break;
          case 1: polyExpr = "[0] + [1]*x"; break;
          case 2: polyExpr = "[0] + [1]*x + [2]*x*x"; break;
          default: polyExpr = "[0]"; break;
        }
        TF1* poly = new TF1(Form("poly_w%d_d%d", w, (int)d), polyExpr, localFitMin, localFitMax);
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
        tex.DrawLatex(0.52, 0.87, "BACKWARD");
        tex.SetTextColor(kBlack);
      } else if (result.phaseTag == 2) {
        tex.SetTextColor(kMagenta+1);
        tex.DrawLatex(0.52, 0.87, "DOUBLING");
        tex.SetTextColor(kBlack);
      }
      if (result.success && result.bkgParams.size() >= 3) {
        tex.SetTextColor(kBlue);
        tex.DrawLatex(0.52, 0.79, Form("#mu_{sig}=%.0f", result.mean));
        tex.DrawLatex(0.52, 0.72, Form("#sigma_{sig}=%.0f", result.sigma));
        tex.SetTextColor(kOrange+1);
        tex.DrawLatex(0.52, 0.65, Form("#sigma_{bkg}=%.0f", result.bkgParams[2]));
        tex.SetTextColor(kBlack);
        tex.DrawLatex(0.52, 0.58, Form("pol%d", result.polyOrder));
        // Color chi2/ndf based on quality
        if (result.chi2ndf >= 0.5 && result.chi2ndf <= 3.0) {
          tex.SetTextColor(kGreen+2);
        } else if (result.chi2ndf > 3.0 && result.chi2ndf <= 10.0) {
          tex.SetTextColor(kOrange+1);
        } else {
          tex.SetTextColor(kRed);
        }
        tex.DrawLatex(0.52, 0.51, Form("#chi^{2}/n=%.1f", result.chi2ndf));
        tex.SetTextColor(kBlack);
      } else {
        tex.SetTextColor(kRed);
        tex.DrawLatex(0.52, 0.77, "Fit failed");
        tex.SetTextColor(kBlack);
      }
      tex.DrawLatex(0.52, 0.44, Form("N=%.0f", result.entries));
      
      gPad->Modified();
      gPad->Update();
    }
    cFits[w]->Modified();
    cFits[w]->Update();
  }

  // =====================================================
  // === CONTOUR PLOTS ===
  // =====================================================
  
  // Individual contours in Δm²
  TCanvas* cContoursDM2[nWidths];
  for (int w = 0; w < nWidths; ++w) {
    cContoursDM2[w] = new TCanvas(Form("c_contour_dm2_%d", w),
                                   Form("Pion Selection in #Delta m^{2} - %s", widthLabels[w].Data()), 800, 600);
    cContoursDM2[w]->cd();
    h2DM2->Draw("colz");

    std::vector<double> pPoints[3], dm2Points[3];
    int nSlices = allMomCenters[w].size();
    
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          pPoints[s].push_back(allMomCenters[w][i]);
          dm2Points[s].push_back(allFitResults[w][i].mean + (s+1) * allFitResults[w][i].sigma);
        }
      }
    }
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          pPoints[s].push_back(allMomCenters[w][i]);
          dm2Points[s].push_back(allFitResults[w][i].mean - (s+1) * allFitResults[w][i].sigma);
        }
      }
    }

    int contColors[3] = {kBlue, kGreen+2, kRed};
    TGraph* graphs[3] = {nullptr};
    for (int s = 0; s < 3; ++s) {
      if (!pPoints[s].empty()) {
        graphs[s] = new TGraph(pPoints[s].size(), pPoints[s].data(), dm2Points[s].data());
        graphs[s]->SetLineColor(contColors[s]);
        graphs[s]->SetLineWidth(4);
        graphs[s]->Draw("L");
      }
    }

    TLine* zeroLine = new TLine(0, 0, 1400, 0);
    zeroLine->SetLineColor(kBlack);
    zeroLine->SetLineStyle(kDashed);
    zeroLine->SetLineWidth(2);
    zeroLine->Draw("same");

    TLine* transLine = new TLine(transitionMom, -30000, transitionMom, 50000);
    transLine->SetLineColor(kMagenta);
    transLine->SetLineStyle(kDashed);
    transLine->SetLineWidth(2);
    transLine->Draw("same");

    TLegend* leg = new TLegend(0.60, 0.60, 0.88, 0.88);
    leg->SetHeader(Form("Width: %s", widthLabels[w].Data()));
    if (graphs[0]) leg->AddEntry(graphs[0], "1#sigma", "l");
    if (graphs[1]) leg->AddEntry(graphs[1], "2#sigma", "l");
    if (graphs[2]) leg->AddEntry(graphs[2], "3#sigma", "l");
    leg->AddEntry(zeroLine, "#Delta m^{2}=0", "l");
    leg->AddEntry(transLine, Form("p=%g (doubling)", transitionMom), "l");
    leg->Draw();
    cContoursDM2[w]->Modified();
    cContoursDM2[w]->Update();
  }

  // Combined 1σ in Δm²
  TCanvas* cCombDM2_1sig = new TCanvas("c_comb_dm2_1sig", "Combined 1#sigma in #Delta m^{2}", 1000, 800);
  cCombDM2_1sig->cd();
  h2DM2->Draw("colz");
  
  TLine* zeroLine1 = new TLine(0, 0, 1400, 0);
  zeroLine1->SetLineColor(kBlack);
  zeroLine1->SetLineStyle(kDashed);
  zeroLine1->SetLineWidth(2);
  zeroLine1->Draw("same");
  
  TLine* transLine1 = new TLine(transitionMom, -30000, transitionMom, 50000);
  transLine1->SetLineColor(kMagenta);
  transLine1->SetLineStyle(kDashed);
  transLine1->SetLineWidth(2);
  transLine1->Draw("same");
  
  TLegend* legDM2_1 = new TLegend(0.60, 0.55, 0.88, 0.88);
  legDM2_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    // Collect raw data
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
    
    // Apply robust smoothing
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    // Build contour from smoothed values
    std::vector<double> pPoints, dm2Points;
    for (size_t i = 0; i < rawP.size(); ++i) {
      pPoints.push_back(rawP[i]);
      dm2Points.push_back(smoothMean[i] + 1.0 * smoothSigma[i]);
    }
    for (int i = (int)rawP.size() - 1; i >= 0; --i) {
      pPoints.push_back(rawP[i]);
      dm2Points.push_back(smoothMean[i] - 1.0 * smoothSigma[i]);
    }
    
    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), dm2Points.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(4);
      g->Draw("L");
      legDM2_1->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legDM2_1->AddEntry(zeroLine1, "#Delta m^{2}=0", "l");
  legDM2_1->AddEntry(transLine1, Form("p=%g", transitionMom), "l");
  legDM2_1->Draw();
  cCombDM2_1sig->Modified();
  cCombDM2_1sig->Update();

  // Combined 3σ in Δm²
  TCanvas* cCombDM2_3sig = new TCanvas("c_comb_dm2_3sig", "Combined 3#sigma in #Delta m^{2}", 1000, 800);
  cCombDM2_3sig->cd();
  h2DM2->Draw("colz");
  
  TLine* zeroLine3 = new TLine(0, 0, 1400, 0);
  zeroLine3->SetLineColor(kBlack);
  zeroLine3->SetLineStyle(kDashed);
  zeroLine3->SetLineWidth(2);
  zeroLine3->Draw("same");
  
  TLine* transLine3 = new TLine(transitionMom, -30000, transitionMom, 50000);
  transLine3->SetLineColor(kMagenta);
  transLine3->SetLineStyle(kDashed);
  transLine3->SetLineWidth(2);
  transLine3->Draw("same");
  
  TLegend* legDM2_3 = new TLegend(0.60, 0.55, 0.88, 0.88);
  legDM2_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    // Collect raw data
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
    
    // Apply robust smoothing
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    // Build contour from smoothed values
    std::vector<double> pPoints, dm2Points;
    for (size_t i = 0; i < rawP.size(); ++i) {
      pPoints.push_back(rawP[i]);
      dm2Points.push_back(smoothMean[i] + 3.0 * smoothSigma[i]);
    }
    for (int i = (int)rawP.size() - 1; i >= 0; --i) {
      pPoints.push_back(rawP[i]);
      dm2Points.push_back(smoothMean[i] - 3.0 * smoothSigma[i]);
    }
    
    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), dm2Points.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(4);
      g->Draw("L");
      legDM2_3->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legDM2_3->AddEntry(zeroLine3, "#Delta m^{2}=0", "l");
  legDM2_3->AddEntry(transLine3, Form("p=%g", transitionMom), "l");
  legDM2_3->Draw();
  cCombDM2_3sig->Modified();
  cCombDM2_3sig->Update();

  // =====================================================
  // === TRANSFORM TO (p, β) SPACE ===
  // =====================================================
  
  TCanvas* cCombPB_1sig = new TCanvas("c_comb_pb_1sig", "Combined 1#sigma in (p, #beta)", 1000, 800);
  cCombPB_1sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB = new TF1("pionCurvePB", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB->SetLineColor(kBlack);
  pionCurvePB->SetLineStyle(kDashed);
  pionCurvePB->SetLineWidth(2);
  pionCurvePB->Draw("same");
  
  TLine* transLinePB1 = new TLine(transitionMom, 0.4, transitionMom, 1.15);
  transLinePB1->SetLineColor(kMagenta);
  transLinePB1->SetLineStyle(kDashed);
  transLinePB1->SetLineWidth(2);
  transLinePB1->Draw("same");
  
  TLegend* legPB_1 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legPB_1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    // First, collect all successful fit results
    std::vector<double> rawP, rawMean, rawSigma;
    int nSlices = allMomCenters[w].size();
    
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        rawP.push_back(allMomCenters[w][i]);
        rawMean.push_back(allFitResults[w][i].mean);
        rawSigma.push_back(allFitResults[w][i].sigma);
      }
    }
    
    if (rawP.size() < 5) continue;  // Need enough points
    
    // Apply robust smoothing (median + Gaussian)
    // Window sizes scale with base width (more smoothing for narrower windows)
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;  // More smoothing for 1x width
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, betaPointsUpper;  // mean + σ → lower β
    std::vector<double> pPointsLower, betaPointsLower;  // mean - σ → higher β
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      
      // Upper boundary in Δm² space → Lower β
      double dm2_upper = smoothMean[i] + 1.0 * smoothSigma[i];
      double beta_upper;
      if (deltaM2ToBeta(p, dm2_upper, beta_upper) && beta_upper > 0.3 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      // Lower boundary in Δm² space → Higher β
      double dm2_lower = smoothMean[i] - 1.0 * smoothSigma[i];
      double beta_lower;
      if (deltaM2ToBeta(p, dm2_lower, beta_lower) && beta_lower > 0.3 && beta_lower < 1.15) {
        pPointsLower.push_back(p);
        betaPointsLower.push_back(beta_lower);
      }
    }

    // Draw both boundaries
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
  legPB_1->AddEntry(transLinePB1, Form("p=%g", transitionMom), "l");
  legPB_1->Draw();
  cCombPB_1sig->Modified();
  cCombPB_1sig->Update();

  // Combined 3σ in (p, β)
  TCanvas* cCombPB_3sig = new TCanvas("c_comb_pb_3sig", "Combined 3#sigma in (p, #beta)", 1000, 800);
  cCombPB_3sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* pionCurvePB3 = new TF1("pionCurvePB3", "x/sqrt(x*x + 139.57*139.57)", 0, 1400);
  pionCurvePB3->SetLineColor(kBlack);
  pionCurvePB3->SetLineStyle(kDashed);
  pionCurvePB3->SetLineWidth(2);
  pionCurvePB3->Draw("same");
  
  TLine* transLinePB3 = new TLine(transitionMom, 0.4, transitionMom, 1.15);
  transLinePB3->SetLineColor(kMagenta);
  transLinePB3->SetLineStyle(kDashed);
  transLinePB3->SetLineWidth(2);
  transLinePB3->Draw("same");
  
  TLegend* legPB_3 = new TLegend(0.12, 0.60, 0.42, 0.88);
  legPB_3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    // First, collect all successful fit results
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
    
    // Apply robust smoothing
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> pPointsUpper, betaPointsUpper;  // mean + 3σ → lower β
    std::vector<double> pPointsLower, betaPointsLower;  // mean - 3σ → higher β
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      
      // Upper boundary in Δm² space → Lower β
      double dm2_upper = smoothMean[i] + 3.0 * smoothSigma[i];
      double beta_upper;
      if (deltaM2ToBeta(p, dm2_upper, beta_upper) && beta_upper > 0.3 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      // Lower boundary in Δm² space → Higher β
      double dm2_lower = smoothMean[i] - 3.0 * smoothSigma[i];
      double beta_lower;
      if (deltaM2ToBeta(p, dm2_lower, beta_lower) && beta_lower > 0.3 && beta_lower < 1.15) {
        pPointsLower.push_back(p);
        betaPointsLower.push_back(beta_lower);
      }
    }

    // Draw both boundaries
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
  legPB_3->AddEntry(transLinePB3, Form("p=%g", transitionMom), "l");
  legPB_3->Draw();
  cCombPB_3sig->Modified();
  cCombPB_3sig->Update();

  // =====================================================
  // === TRANSFORM TO (mass, a) SPACE ===
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
    // Collect raw data
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
    
    // Apply robust smoothing
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> massPointsUpper, aPointsUpper;  // mean + σ → higher mass
    std::vector<double> massPointsLower, aPointsLower;  // mean - σ → lower mass
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      
      // Upper boundary in Δm² → higher mass
      double dm2_upper = smoothMean[i] + 1.0 * smoothSigma[i];
      double beta_u, mass_u, a_u;
      if (deltaM2ToBeta(p, dm2_upper, beta_u) && beta_u > 0.1 && beta_u < 1.5) {
        if (pBetaToMassA(p, beta_u, gSSquared, mass_u, a_u) && mass_u > 0 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      // Lower boundary in Δm² → lower mass
      double dm2_lower = smoothMean[i] - 1.0 * smoothSigma[i];
      double beta_l, mass_l, a_l;
      if (deltaM2ToBeta(p, dm2_lower, beta_l) && beta_l > 0.1 && beta_l < 1.5) {
        if (pBetaToMassA(p, beta_l, gSSquared, mass_l, a_l) && mass_l > 0 && mass_l < 300 && a_l > 0 && a_l < 16) {
          massPointsLower.push_back(mass_l);
          aPointsLower.push_back(a_l);
        }
      }
    }

    // Draw both boundaries
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

  // Combined 3σ in (mass, a)
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
    // Collect raw data
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
    
    // Apply robust smoothing
    int medWin = (w == 0) ? 7 : (w == 1) ? 5 : 3;
    int gaussWin = (w == 0) ? 9 : (w == 1) ? 7 : 5;
    std::vector<double> smoothMean = robustSmooth(rawMean, medWin, gaussWin);
    std::vector<double> smoothSigma = robustSmooth(rawSigma, medWin, gaussWin);
    
    std::vector<double> massPointsUpper, aPointsUpper;  // mean + 3σ → higher mass
    std::vector<double> massPointsLower, aPointsLower;  // mean - 3σ → lower mass
    
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      
      // Upper boundary in Δm² → higher mass
      double dm2_upper = smoothMean[i] + 3.0 * smoothSigma[i];
      double beta_u, mass_u, a_u;
      if (deltaM2ToBeta(p, dm2_upper, beta_u) && beta_u > 0.1 && beta_u < 1.5) {
        if (pBetaToMassA(p, beta_u, gSSquared, mass_u, a_u) && mass_u > 0 && mass_u < 300 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      // Lower boundary in Δm² → lower mass
      double dm2_lower = smoothMean[i] - 3.0 * smoothSigma[i];
      double beta_l, mass_l, a_l;
      if (deltaM2ToBeta(p, dm2_lower, beta_l) && beta_l > 0.1 && beta_l < 1.5) {
        if (pBetaToMassA(p, beta_l, gSSquared, mass_l, a_l) && mass_l > 0 && mass_l < 300 && a_l > 0 && a_l < 16) {
          massPointsLower.push_back(mass_l);
          aPointsLower.push_back(a_l);
        }
      }
    }

    // Draw both boundaries
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
  // === PARAMETER PLOTS ===
  // =====================================================
  
  TCanvas* cPar = new TCanvas("c_par", "Fit Parameters vs Momentum", 1000, 450);
  cPar->Divide(2, 1);

  cPar->cd(1);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgMean = new TMultiGraph();
  TLegend* legMean = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allMomCenters[w].size(); ++i)
      if (allFitResults[w][i].success)
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].mean);
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.5);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgMean->Add(g, "LP");
      legMean->AddEntry(g, Form("%s", widthLabels[w].Data()), "lp");
    }
  }
  mgMean->SetTitle("#Delta m^{2} Mean vs Momentum;Momentum [MeV/c];#Delta m^{2} Mean [MeV^{2}/c^{4}]");
  mgMean->Draw("A");
  mgMean->GetXaxis()->SetLimits(0.0, 1400.0);
  mgMean->GetYaxis()->SetRangeUser(-5000.0, 5000.0);
  
  TLine* zeroLineMean = new TLine(0, 0, 1400, 0);
  zeroLineMean->SetLineColor(kBlack);
  zeroLineMean->SetLineStyle(kDashed);
  zeroLineMean->Draw("same");
  
  TLine* transLineMean = new TLine(transitionMom, -5000, transitionMom, 5000);
  transLineMean->SetLineColor(kMagenta);
  transLineMean->SetLineStyle(kDashed);
  transLineMean->Draw("same");
  
  legMean->AddEntry(zeroLineMean, "#Delta m^{2}=0", "l");
  legMean->Draw();

  cPar->cd(2);
  gPad->SetLeftMargin(0.14);
  TMultiGraph* mgSigma = new TMultiGraph();
  TLegend* legSigma = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allMomCenters[w].size(); ++i)
      if (allFitResults[w][i].success)
        g->SetPoint(np++, allMomCenters[w][i], allFitResults[w][i].sigma);
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.5);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgSigma->Add(g, "LP");
      legSigma->AddEntry(g, Form("%s", widthLabels[w].Data()), "lp");
    }
  }
  mgSigma->SetTitle("#Delta m^{2} Width vs Momentum;Momentum [MeV/c];#sigma [MeV^{2}/c^{4}]");
  mgSigma->Draw("A");
  mgSigma->GetXaxis()->SetLimits(0.0, 1400.0);
  mgSigma->GetYaxis()->SetRangeUser(0.0, 25000.0);
  
  TLine* transLineSigma = new TLine(transitionMom, 0, transitionMom, 25000);
  transLineSigma->SetLineColor(kMagenta);
  transLineSigma->SetLineStyle(kDashed);
  transLineSigma->Draw("same");
  
  legSigma->Draw();
  cPar->Modified();
  cPar->Update();

  // =====================================================
  // === SAVE ALL ===
  // =====================================================
  
  for (int w = 0; w < nWidths; ++w) {
    cFits[w]->SaveAs(Form("pion_fits_dm2_%d.png", w));
    cContoursDM2[w]->SaveAs(Form("pion_contours_dm2_%d.png", w));
  }
  cCombDM2_1sig->SaveAs("pion_dm2_combined_1sigma.png");
  cCombDM2_3sig->SaveAs("pion_dm2_combined_3sigma.png");
  cCombPB_1sig->SaveAs("pion_pbeta_combined_1sigma.png");
  cCombPB_3sig->SaveAs("pion_pbeta_combined_3sigma.png");
  cCombMA_1sig->SaveAs("pion_massa_combined_1sigma.png");
  cCombMA_3sig->SaveAs("pion_massa_combined_3sigma.png");
  cPar->SaveAs("pion_parameters_vs_momentum.png");

  // Output file
  std::ofstream outfile("pion_fit_results_dm2.txt");
  outfile << "Width\tpCenter\tpLow\tpHigh\tSliceWidth\tPhase\tMean_dm2\tSigma_sig\tSigma_bkg\tPolyOrder\tChi2NDF\tStatus" << std::endl;
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allMomCenters[w].size(); ++i) {
      FitResult& r = allFitResults[w][i];
      TString phaseStr = (r.phaseTag == 0) ? "BWD" : ((r.phaseTag == 1) ? "FWD" : "DBL");
      double sigmaBkg = (r.bkgParams.size() >= 3) ? r.bkgParams[2] : 0.0;
      outfile << widthLabels[w] << "\t" << r.pCenter << "\t" << r.pLow << "\t" << r.pHigh << "\t"
              << r.sliceWidth << "\t" << phaseStr << "\t"
              << r.mean << "\t" << r.sigma << "\t" << sigmaBkg << "\t" 
              << r.polyOrder << "\t" << r.chi2ndf << "\t"
              << (r.success ? "OK" : "FAIL") << std::endl;
    }
  }
  outfile.close();

  cout << "\n=== Analysis Complete ===" << endl;
  cout << "FIT MODEL: Signal Gauss + Background Gauss + Polynomial" << endl;
  cout << "  - TWO-PASS FITTING to anchor signal to data peak:" << endl;
  cout << "    Pass 1: Fix signal mean to histogram peak, fit other params" << endl;
  cout << "    Pass 2: Allow signal mean ±2000 variation around peak" << endl;
  cout << "  - Signal Gauss: narrow (σ = 1500-10000), mean near data peak" << endl;
  cout << "  - Background Gauss: broad (σ = 12000-45000), free position" << endl;
  cout << "  - Polynomial: pol0-pol2" << endl;
  cout << "  - Fit range: [-15000, 30000] for p < 300 MeV/c" << endl;
  cout << "               [-20000, 50000] for p >= 300 MeV/c" << endl;
  cout << "\nTHREE-PHASE ADAPTIVE WIDTH APPROACH:" << endl;
  cout << "  Phase 0 (backward): p < " << startMom << " MeV/c, right edge anchored, extend left" << endl;
  cout << "  Phase 1 (forward):  " << startMom << " <= p < " << transitionMom << " MeV/c, fixed width sliding" << endl;
  cout << "  Phase 2 (doubling): p >= " << transitionMom << " MeV/c, width doubles each step" << endl;
  cout << "\nBase widths: 5, 10, 20, 40 MeV/c" << endl;
  cout << "Start momentum: " << startMom << " MeV/c" << endl;
  cout << "Transition at: " << transitionMom << " MeV/c" << endl;
  cout << "\nOutput files:" << endl;
  cout << "  - pion_fits_dm2_*.png" << endl;
  cout << "  - pion_contours_dm2_*.png" << endl;
  cout << "  - pion_dm2_combined_1sigma.png, _3sigma.png" << endl;
  cout << "  - pion_pbeta_combined_1sigma.png, _3sigma.png" << endl;
  cout << "  - pion_massa_combined_1sigma.png, _3sigma.png" << endl;
  cout << "  - pion_parameters_vs_momentum.png" << endl;
  cout << "  - pion_fit_results_dm2.txt" << endl;
}
