// =====================================================
// pid_systematics_calculator_pip.C
// =====================================================
// 
// PURPOSE: Calculate PID selection systematic uncertainties
//          by comparing cuts from different data samples,
//          run periods, or analysis hypotheses.
//
// KEY DESIGN: Simulation is shown for comparison but EXCLUDED from systematics!
//
// SIGMA LEVELS: 1, 2.5, 3, 3.5, 5 sigma envelope cuts saved
//
// USAGE:
//   root -l 'pid_systematics_calculator_pip.C("pip", "pip_syst_output.root")'
//
// =====================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TSpline.h"
#include "TCutG.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include "TF1.h"
#include "TAxis.h"
#include "TColor.h"
#include "TNamed.h"
#include "TSystem.h"

using std::cout; using std::endl;
using std::vector; using std::map; using std::string;

// =====================================================
// GLOBAL CONFIGURATION - PION+ (pip)
// =====================================================
const double gMomentumMax = 2000.0;  // Pion momentum range

// =====================================================
// DATA STRUCTURES
// =====================================================

enum SampleType {
  kExperimental = 0,   // Real data - INCLUDED in systematics
  kSimulation   = 1,   // MC - shown for comparison, EXCLUDED from systematics
  kReference    = 2    // Your best-judgment reference (also experimental)
};

struct SampleInfo {
  TString filename;
  TString label;
  TString category;
  SampleType type;
  int color;
  int markerStyle;
  int lineStyle;
};

struct FitPoint {
  double p, mean, sigma, chi2ndf;
  bool valid;
};

struct ComparisonPoint {
  double p;
  double meanAvg, meanRMS, meanMin, meanMax;
  double sigmaAvg, sigmaRMS, sigmaMin, sigmaMax;
  int nExpSamples;
  double meanRef, sigmaRef;
  bool hasRef;
  vector<double> meanSim, sigmaSim;
  vector<TString> simLabels;
};

// =====================================================
// HELPER FUNCTIONS
// =====================================================

double calcRMS(const vector<double>& v) {
  if (v.size() < 2) return 0;
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double sq_sum = 0;
  for (double x : v) sq_sum += (x - mean) * (x - mean);
  return std::sqrt(sq_sum / (v.size() - 1));
}

double calcMean(const vector<double>& v) {
  if (v.empty()) return 0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// =====================================================
// SYSTEMATIC YIELD CALCULATOR
// =====================================================

struct SystematicResult {
  double p;
  double efficiency_nominal;
  double efficiency_envelope;
  double delta_efficiency;
  double relative_systematic;
};

double gaussianCDF(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double gaussianPDF(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double calcEfficiency(double mu_true, double sigma_true, 
                      double mu_cut, double sigma_cut, double Nsigma) {
  double lower = (mu_cut - Nsigma * sigma_cut - mu_true) / sigma_true;
  double upper = (mu_cut + Nsigma * sigma_cut - mu_true) / sigma_true;
  return gaussianCDF(upper) - gaussianCDF(lower);
}

vector<SystematicResult> calculateYieldSystematics(
    const vector<ComparisonPoint>& comparison,
    double Nsigma = 3.0
) {
  vector<SystematicResult> results;
  
  cout << "\n" << string(80, '=') << endl;
  cout << "  YIELD SYSTEMATIC CALCULATION (N=" << Nsigma << " sigma cut)" << endl;
  cout << string(80, '=') << endl;
  
  cout << "\nTheory: For Gaussian signal with ±" << Nsigma << "σ cut:" << endl;
  double nominalEff = 2.0 * gaussianCDF(Nsigma) - 1.0;
  cout << "  Nominal efficiency = " << Form("%.6f", nominalEff) 
       << " (" << Form("%.4f%%", 100*(1-nominalEff)) << " loss)" << endl;
  cout << "  Gaussian PDF at " << Nsigma << "σ = " << Form("%.6f", gaussianPDF(Nsigma)) << endl;
  cout << "\n  The 'scary' 20% σ uncertainty translates to efficiency uncertainty of:" << endl;
  double dEff_approx = 2.0 * gaussianPDF(Nsigma) * Nsigma * 0.20;
  cout << "  δε ≈ 2 × G(" << Nsigma << ") × " << Nsigma << " × 0.20 = " 
       << Form("%.5f", dEff_approx) << " = " << Form("%.3f%%", 100*dEff_approx) << endl;
  
  cout << "\n" << string(80, '-') << endl;
  cout << Form("%-10s %12s %12s %12s %12s %12s", 
               "p [MeV/c]", "σ_avg", "σ_RMS", "ε_nominal", "ε_envelope", "δε [%]") << endl;
  cout << string(80, '-') << endl;
  
  double totalDeltaEff = 0;
  int nPoints = 0;
  
  for (const auto& cp : comparison) {
    if (cp.nExpSamples < 2) continue;
    
    SystematicResult sr;
    sr.p = cp.p;
    sr.efficiency_nominal = 2.0 * gaussianCDF(Nsigma) - 1.0;
    
    double sigma_ratio = cp.sigmaMax / cp.sigmaAvg;
    sr.efficiency_envelope = calcEfficiency(0, 1.0, 0, sigma_ratio, Nsigma);
    
    double sigma_ratio_min = cp.sigmaMin / cp.sigmaAvg;
    double eff_narrow = calcEfficiency(0, 1.0, 0, sigma_ratio_min, Nsigma);
    
    sr.delta_efficiency = std::abs(sr.efficiency_envelope - eff_narrow) / 2.0;
    
    double delta_eff_analytical = 2.0 * gaussianPDF(Nsigma) * 
                                  std::sqrt(Nsigma*Nsigma * std::pow(cp.sigmaRMS/cp.sigmaAvg, 2) +
                                           std::pow(cp.meanRMS/cp.sigmaAvg, 2));
    
    sr.delta_efficiency = std::max(sr.delta_efficiency, delta_eff_analytical);
    sr.relative_systematic = 100.0 * sr.delta_efficiency / sr.efficiency_nominal;
    
    results.push_back(sr);
    totalDeltaEff += sr.delta_efficiency;
    nPoints++;
    
    if (((int)cp.p % 100) < 15) {
      cout << Form("%-10.0f %12.5f %12.5f %12.6f %12.6f %12.4f",
                   cp.p, cp.sigmaAvg, cp.sigmaRMS, 
                   sr.efficiency_nominal, sr.efficiency_envelope,
                   sr.relative_systematic) << endl;
    }
  }
  
  cout << string(80, '-') << endl;
  
  if (nPoints > 0) {
    double avgDeltaEff = totalDeltaEff / nPoints;
    cout << "\nSUMMARY for " << Nsigma << "σ cut:" << endl;
    cout << Form("  Average efficiency uncertainty: %.5f (%.4f%%)", 
                 avgDeltaEff, 100*avgDeltaEff/nominalEff) << endl;
    cout << Form("  This means: for 1000 signal events, uncertainty is ±%.1f events", 
                 1000*avgDeltaEff) << endl;
  }
  
  cout << "\n" << string(80, '=') << endl;
  cout << "RECOMMENDATION:" << endl;
  cout << "  1. Use REFERENCE TCutG (Run 049) as your nominal selection" << endl;
  cout << "  2. For systematic: vary using envelope cuts, quote yield difference" << endl;
  cout << "  3. Alternative: quote " << Form("%.2f%%", 100*totalDeltaEff/nPoints/nominalEff) 
       << " as flat efficiency systematic" << endl;
  cout << string(80, '=') << endl;
  
  return results;
}

// =====================================================
// TCUTG COMPARISON PLOT (without zoomed version)
// =====================================================

void plotTCutGComparison(
    const TString& expFile,
    const TString& simFile,
    const TString& particleName,
    const TString& outputPrefix,
    int widthIdx = 1
) {
  cout << "\n" << string(80, '=') << endl;
  cout << "  CREATING TCutG COMPARISON PLOT" << endl;
  cout << string(80, '=') << endl;
  
  TString cut3exp_name = Form("cut_%s_3sig_w%d", particleName.Data(), widthIdx);
  TString cut5exp_name = Form("cut_%s_5sig_w%d", particleName.Data(), widthIdx);
  
  TFile* fExp = TFile::Open(expFile, "READ");
  TCutG* cut3exp = nullptr;
  TCutG* cut5exp = nullptr;
  
  if (fExp && !fExp->IsZombie()) {
    cut3exp = (TCutG*)fExp->Get(Form("TCutG/%s", cut3exp_name.Data()));
    cut5exp = (TCutG*)fExp->Get(Form("TCutG/%s", cut5exp_name.Data()));
    if (cut3exp) cut3exp = (TCutG*)cut3exp->Clone("cut3exp");
    if (cut5exp) cut5exp = (TCutG*)cut5exp->Clone("cut5exp");
    fExp->Close();
  }
  
  TString cut3sim_name = Form("cut_%s_3sig_w%d", particleName.Data(), widthIdx);
  TString cut5sim_name = Form("cut_%s_5sig_w%d", particleName.Data(), widthIdx);
  
  TFile* fSim = TFile::Open(simFile, "READ");
  TCutG* cut3sim = nullptr;
  TCutG* cut5sim = nullptr;
  
  if (fSim && !fSim->IsZombie()) {
    cut3sim = (TCutG*)fSim->Get(Form("TCutG/%s", cut3sim_name.Data()));
    cut5sim = (TCutG*)fSim->Get(Form("TCutG/%s", cut5sim_name.Data()));
    if (cut3sim) cut3sim = (TCutG*)cut3sim->Clone("cut3sim");
    if (cut5sim) cut5sim = (TCutG*)cut5sim->Clone("cut5sim");
    fSim->Close();
  }
  
  cout << "  Loaded from " << expFile << ": " 
       << (cut3exp ? "3σ OK" : "3σ MISSING") << ", "
       << (cut5exp ? "5σ OK" : "5σ MISSING") << endl;
  cout << "  Loaded from " << simFile << ": "
       << (cut3sim ? "3σ OK" : "3σ MISSING") << ", "
       << (cut5sim ? "5σ OK" : "5σ MISSING") << endl;
  
  // Pion mass
  double mass = 139.57;
  TF1* fTheory = new TF1("fTheory", "x/sqrt(x*x + [0]*[0])", 0, gMomentumMax);
  fTheory->SetParameter(0, mass);
  fTheory->SetLineColor(kGray+1);
  fTheory->SetLineStyle(2);
  fTheory->SetLineWidth(2);
  
  TCanvas* cComp = new TCanvas("c_tcutg_comparison", "TCutG Comparison: Exp vs Sim", 1400, 600);
  cComp->Divide(2, 1);
  
  // ===== Left panel: 3-sigma cuts =====
  cComp->cd(1);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.08);
  gPad->SetBottomMargin(0.12);
  
  // Y-axis range changed to 0-1.5
  TH2F* frame3 = new TH2F("frame3", Form("%s: 3#sigma TCutG Comparison;p [MeV/c];#beta", particleName.Data()),
                          100, 0, gMomentumMax, 100, 0.0, 1.5);
  frame3->SetStats(0);
  frame3->Draw();
  
  fTheory->Draw("same");
  
  if (cut3exp) {
    cut3exp->SetLineColor(kRed);
    cut3exp->SetLineWidth(3);
    cut3exp->SetLineStyle(1);
    cut3exp->SetFillStyle(0);
    cut3exp->Draw("L same");
  }
  
  if (cut3sim) {
    cut3sim->SetLineColor(kBlue);
    cut3sim->SetLineWidth(3);
    cut3sim->SetLineStyle(2);
    cut3sim->SetFillStyle(0);
    cut3sim->Draw("L same");
  }
  
  TLegend* leg3 = new TLegend(0.50, 0.15, 0.92, 0.35);
  leg3->SetTextSize(0.035);
  leg3->SetBorderSize(0);
  leg3->SetFillStyle(0);
  if (cut3exp) leg3->AddEntry(cut3exp, "Exp (Run 049) 3#sigma", "l");
  if (cut3sim) leg3->AddEntry(cut3sim, "Sim (SMASH) 3#sigma", "l");
  leg3->AddEntry(fTheory, "#beta_{theory}(p)", "l");
  leg3->Draw();
  
  TLatex tex3;
  tex3.SetNDC();
  tex3.SetTextSize(0.032);
  tex3.SetTextColor(kBlue+2);
  tex3.DrawLatex(0.15, 0.88, "#pi^{+}: 3#sigma comparison");
  
  // ===== Right panel: 5-sigma cuts =====
  cComp->cd(2);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.08);
  gPad->SetBottomMargin(0.12);
  
  // Y-axis range changed to 0-1.5
  TH2F* frame5 = new TH2F("frame5", Form("%s: 5#sigma TCutG Comparison;p [MeV/c];#beta", particleName.Data()),
                          100, 0, gMomentumMax, 100, 0.0, 1.5);
  frame5->SetStats(0);
  frame5->Draw();
  
  TF1* fTheory2 = (TF1*)fTheory->Clone("fTheory2");
  fTheory2->Draw("same");
  
  if (cut5exp) {
    cut5exp->SetLineColor(kRed);
    cut5exp->SetLineWidth(3);
    cut5exp->SetLineStyle(1);
    cut5exp->SetFillStyle(0);
    cut5exp->Draw("L same");
  }
  
  if (cut5sim) {
    cut5sim->SetLineColor(kBlue);
    cut5sim->SetLineWidth(3);
    cut5sim->SetLineStyle(2);
    cut5sim->SetFillStyle(0);
    cut5sim->Draw("L same");
  }
  
  TLegend* leg5 = new TLegend(0.50, 0.15, 0.92, 0.35);
  leg5->SetTextSize(0.035);
  leg5->SetBorderSize(0);
  leg5->SetFillStyle(0);
  if (cut5exp) leg5->AddEntry(cut5exp, "Exp (Run 049) 5#sigma", "l");
  if (cut5sim) leg5->AddEntry(cut5sim, "Sim (SMASH) 5#sigma", "l");
  leg5->AddEntry(fTheory2, "#beta_{theory}(p)", "l");
  leg5->Draw();
  
  TLatex tex5;
  tex5.SetNDC();
  tex5.SetTextSize(0.032);
  tex5.SetTextColor(kGreen+2);
  tex5.DrawLatex(0.15, 0.88, "#pi^{+}: 5#sigma comparison");
  
  // NOTE: Zoomed canvas removed as requested
}

// =====================================================
// SIMULATION MISMATCH CALCULATOR
// =====================================================

void calculateSimMismatch(
    const vector<ComparisonPoint>& comparison,
    double Nsigma = 3.0
) {
  cout << "\n" << string(80, '=') << endl;
  cout << "  DATA-SIMULATION CUT MISMATCH ANALYSIS" << endl;
  cout << string(80, '=') << endl;
  
  cout << "\nScenario: Using EXPERIMENTAL TCutG on SIMULATION data" << endl;
  cout << "(or equivalently: simulation events evaluated against exp cuts)\n" << endl;
  
  bool hasSim = false;
  for (const auto& cp : comparison) {
    if (!cp.meanSim.empty()) { hasSim = true; break; }
  }
  
  if (!hasSim) {
    cout << "[INFO] No simulation data available for comparison." << endl;
    return;
  }
  
  cout << "For each momentum, we calculate:" << endl;
  cout << "  - ε_matched: efficiency if sim uses its own cut (=nominal ~99.73% for 3σ)" << endl;
  cout << "  - ε_mismatched: efficiency if sim is selected with exp cut" << endl;
  cout << "  - Δε: the efficiency loss/gain due to mismatch" << endl;
  
  cout << "\n" << string(80, '-') << endl;
  cout << Form("%-10s %10s %10s %10s %10s %12s", 
               "p [MeV/c]", "μ_exp", "μ_sim", "σ_exp", "σ_sim", "Δε [%]") << endl;
  cout << string(80, '-') << endl;
  
  double totalMismatch = 0;
  int nPoints = 0;
  double maxMismatch = 0;
  double pAtMaxMismatch = 0;
  
  for (const auto& cp : comparison) {
    if (cp.meanSim.empty()) continue;
    
    double mu_exp = cp.meanAvg;
    double sigma_exp = cp.sigmaAvg;
    double mu_sim = cp.meanSim[0];
    double sigma_sim = cp.sigmaSim[0];
    
    double eff_matched = 2.0 * gaussianCDF(Nsigma) - 1.0;
    double eff_mismatched = calcEfficiency(mu_sim, sigma_sim, mu_exp, sigma_exp, Nsigma);
    
    double delta_eff = eff_mismatched - eff_matched;
    double delta_eff_percent = 100.0 * delta_eff;
    
    totalMismatch += std::abs(delta_eff);
    nPoints++;
    
    if (std::abs(delta_eff) > maxMismatch) {
      maxMismatch = std::abs(delta_eff);
      pAtMaxMismatch = cp.p;
    }
    
    if (((int)cp.p % 100) < 15) {
      cout << Form("%-10.0f %10.5f %10.5f %10.5f %10.5f %+12.4f",
                   cp.p, mu_exp, mu_sim, sigma_exp, sigma_sim, delta_eff_percent) << endl;
    }
  }
  
  cout << string(80, '-') << endl;
  
  if (nPoints > 0) {
    double avgMismatch = totalMismatch / nPoints;
    cout << "\nSUMMARY for " << Nsigma << "σ cut:" << endl;
    cout << Form("  Average |Δε|: %.5f (%.4f%%)", avgMismatch, 100*avgMismatch) << endl;
    cout << Form("  Maximum |Δε|: %.5f (%.4f%%) at p=%.0f MeV/c", 
                 maxMismatch, 100*maxMismatch, pAtMaxMismatch) << endl;
    cout << Form("  For 1000 sim events: average ±%.1f events affected", 1000*avgMismatch) << endl;
  }
  
  cout << "\n" << string(80, '=') << endl;
  cout << "INTERPRETATION:" << endl;
  cout << "  Δε > 0: Exp cut is WIDER than needed → slight over-selection" << endl;
  cout << "  Δε < 0: Exp cut is NARROWER than needed → signal loss in simulation" << endl;
  cout << "\nRECOMMENDATION:" << endl;
  cout << "  1. For SIMULATION analysis: use simulation-derived TCutG" << endl;
  cout << "  2. If using exp TCutG on sim: quote " << Form("%.3f%%", 100*totalMismatch/nPoints) 
       << " efficiency correction" << endl;
  cout << "  3. The sign tells you direction: check if sim σ > exp σ" << endl;
  cout << string(80, '=') << endl;
}

// Note: isSimulation=true relaxes chi2 cut
vector<FitPoint> loadFitResults(const TString& filename, int widthIdx = 1, bool debug = false, bool isSimulation = false) {
  vector<FitPoint> results;
  
  TFile* f = TFile::Open(filename, "READ");
  if (!f || f->IsZombie()) {
    cerr << "[ERROR] Cannot open " << filename << endl;
    return results;
  }
  
  TTree* tree = (TTree*)f->Get("FitResults");
  if (!tree) {
    cerr << "[ERROR] No FitResults tree in " << filename << endl;
    f->Close();
    return results;
  }
  
  Double_t p, mean, sigma, chi2;
  Int_t width_idx, success;
  
  tree->SetBranchAddress("p_center", &p);
  tree->SetBranchAddress("delta_beta_mean", &mean);
  tree->SetBranchAddress("delta_beta_sigma", &sigma);
  tree->SetBranchAddress("chi2ndf", &chi2);
  tree->SetBranchAddress("width_idx", &width_idx);
  tree->SetBranchAddress("success", &success);
  
  Long64_t nEntries = tree->GetEntries();
  
  int nTotal = 0, nWrongWidth = 0, nFailed = 0;
  int nBadChi2 = 0, nBadSigma = 0, nBadMean = 0, nPassed = 0;
  std::map<int, int> widthCounts;
  
  double chi2Cut = isSimulation ? 1000.0 : 10.0;
  
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);
    nTotal++;
    widthCounts[width_idx]++;
    
    if (width_idx != widthIdx) { nWrongWidth++; continue; }
    if (success != 1) { nFailed++; continue; }
    if (chi2 > chi2Cut) { 
      if (debug && nBadChi2 < 5) cout << "    [DEBUG] chi2=" << chi2 << " at p=" << p << endl;
      nBadChi2++; continue; 
    }
    if (sigma < 0.003 || sigma > 0.07) { 
      if (debug && nBadSigma < 5) cout << "    [DEBUG] sigma=" << sigma << " at p=" << p << endl;
      nBadSigma++; continue; 
    }
    if (std::abs(mean) > 0.05) { 
      if (debug && nBadMean < 5) cout << "    [DEBUG] mean=" << mean << " at p=" << p << endl;
      nBadMean++; continue; 
    }
    
    nPassed++;
    FitPoint pt = {p, mean, sigma, chi2, true};
    results.push_back(pt);
  }
  
  f->Close();
  std::sort(results.begin(), results.end(), 
            [](const FitPoint& a, const FitPoint& b) { return a.p < b.p; });
  
  cout << "[INFO] Loaded " << results.size() << " points from " << filename << endl;
  
  if (results.size() < 100 || debug) {
    cout << "  [DEBUG] " << filename << ":" << endl;
    cout << "    Total entries: " << nTotal << endl;
    cout << "    Width indices in file: ";
    for (auto& kv : widthCounts) cout << "w" << kv.first << "=" << kv.second << " ";
    cout << endl;
    cout << "    Requested width_idx: " << widthIdx << endl;
    cout << "    Chi2 cut used: " << chi2Cut << (isSimulation ? " (relaxed for sim)" : "") << endl;
    cout << "    Filtered by width_idx: " << nWrongWidth << endl;
    cout << "    Filtered by success!=1: " << nFailed << endl;
    cout << "    Filtered by chi2>" << chi2Cut << ": " << nBadChi2 << endl;
    cout << "    Filtered by sigma out of [0.003,0.07]: " << nBadSigma << endl;
    cout << "    Filtered by |mean|>0.05: " << nBadMean << endl;
    cout << "    PASSED all cuts: " << nPassed << endl;
  }
  
  return results;
}

// =====================================================
// LOAD TCutG FROM FILE
// =====================================================
TCutG* loadTCutG(const TString& filename, const TString& cutName) {
  TFile* f = TFile::Open(filename, "READ");
  if (!f || f->IsZombie()) return nullptr;
  
  TCutG* cut = (TCutG*)f->Get(Form("TCutG/%s", cutName.Data()));
  if (cut) cut = (TCutG*)cut->Clone();
  f->Close();
  return cut;
}

// =====================================================
// COMPARE FIT RESULTS ACROSS SAMPLES
// =====================================================
vector<ComparisonPoint> compareFitResults(
    const vector<SampleInfo>& samples,
    const vector<vector<FitPoint>>& allResults,
    double pStep = 10.0
) {
  vector<ComparisonPoint> comparison;
  if (allResults.empty()) return comparison;
  
  double pMin = 1e9, pMax = -1e9;
  for (const auto& results : allResults) {
    for (const auto& pt : results) {
      if (pt.p < pMin) pMin = pt.p;
      if (pt.p > pMax) pMax = pt.p;
    }
  }
  
  int refIdx = -1;
  for (size_t s = 0; s < samples.size(); ++s) {
    if (samples[s].type == kReference) { refIdx = s; break; }
  }
  
  for (double p = pMin; p <= pMax; p += pStep) {
    vector<double> expMeans, expSigmas, simMeans, simSigmas;
    vector<TString> simLabels;
    double refMean = 0, refSigma = 0;
    bool hasRef = false;
    
    for (size_t s = 0; s < samples.size(); ++s) {
      const FitPoint* bestPt = nullptr;
      double bestDist = 1e9;
      for (const auto& pt : allResults[s]) {
        double dist = std::abs(pt.p - p);
        if (dist < bestDist && dist < pStep) { bestDist = dist; bestPt = &pt; }
      }
      if (!bestPt) continue;
      
      if (samples[s].type == kSimulation) {
        simMeans.push_back(bestPt->mean);
        simSigmas.push_back(bestPt->sigma);
        simLabels.push_back(samples[s].label);
      } else {
        expMeans.push_back(bestPt->mean);
        expSigmas.push_back(bestPt->sigma);
        if ((int)s == refIdx) { refMean = bestPt->mean; refSigma = bestPt->sigma; hasRef = true; }
      }
    }
    
    if (expMeans.empty()) continue;
    
    ComparisonPoint cp;
    cp.p = p;
    cp.nExpSamples = expMeans.size();
    cp.meanAvg = calcMean(expMeans);
    cp.meanRMS = (expMeans.size() >= 2) ? calcRMS(expMeans) : 0;
    cp.meanMin = *std::min_element(expMeans.begin(), expMeans.end());
    cp.meanMax = *std::max_element(expMeans.begin(), expMeans.end());
    cp.sigmaAvg = calcMean(expSigmas);
    cp.sigmaRMS = (expSigmas.size() >= 2) ? calcRMS(expSigmas) : 0;
    cp.sigmaMin = *std::min_element(expSigmas.begin(), expSigmas.end());
    cp.sigmaMax = *std::max_element(expSigmas.begin(), expSigmas.end());
    cp.hasRef = hasRef; cp.meanRef = refMean; cp.sigmaRef = refSigma;
    cp.meanSim = simMeans; cp.sigmaSim = simSigmas; cp.simLabels = simLabels;
    comparison.push_back(cp);
  }
  return comparison;
}

// =====================================================
// GENERATE ENVELOPE CUT
// =====================================================
TCutG* generateEnvelopeCut(const vector<TCutG*>& cuts, const TString& cutName,
                           double particleMass, double pMin, double pMax, double pStep = 5.0) {
  if (cuts.empty()) return nullptr;
  
  vector<double> pPoints, betaUpper, betaLower;
  for (double p = pMin; p <= pMax; p += pStep) {
    double maxBeta = -1e9, minBeta = 1e9;
    for (const auto& cut : cuts) {
      if (!cut) continue;
      int nPts = cut->GetN();
      const Double_t* xPts = cut->GetX();
      const Double_t* yPts = cut->GetY();
      for (int i = 0; i < nPts - 1; ++i) {
        double x1 = xPts[i], x2 = xPts[i+1], y1 = yPts[i], y2 = yPts[i+1];
        if ((x1 <= p && p <= x2) || (x2 <= p && p <= x1)) {
          double t = (x2 != x1) ? (p - x1) / (x2 - x1) : 0.5;
          double beta = y1 + t * (y2 - y1);
          if (beta > maxBeta) maxBeta = beta;
          if (beta < minBeta) minBeta = beta;
        }
      }
    }
    if (maxBeta > minBeta) {
      pPoints.push_back(p); betaUpper.push_back(maxBeta); betaLower.push_back(minBeta);
    }
  }
  
  if (pPoints.empty()) return nullptr;
  
  vector<double> pCut, betaCut;
  for (size_t i = 0; i < pPoints.size(); ++i) { pCut.push_back(pPoints[i]); betaCut.push_back(betaUpper[i]); }
  for (int i = pPoints.size() - 1; i >= 0; --i) { pCut.push_back(pPoints[i]); betaCut.push_back(betaLower[i]); }
  pCut.push_back(pCut[0]); betaCut.push_back(betaCut[0]);
  
  TCutG* envelope = new TCutG(cutName, pCut.size(), pCut.data(), betaCut.data());
  envelope->SetLineColor(kBlack); envelope->SetLineWidth(3); envelope->SetLineStyle(kDashed);
  return envelope;
}

// =====================================================
// CREATE COMPARISON PLOTS
// =====================================================
void createComparisonPlots(const vector<SampleInfo>& samples, const vector<vector<FitPoint>>& allResults,
                           const vector<ComparisonPoint>& comparison, const TString& particleName,
                           const TString& outputPrefix) {
  gStyle->SetOptStat(0);
  
  int refIdx = -1;
  for (size_t s = 0; s < samples.size(); ++s) if (samples[s].type == kReference) { refIdx = s; break; }
  
  // ===== Plot 1: Mean vs momentum =====
  TCanvas* cMean = new TCanvas("c_mean_comparison", "Mean Comparison", 1400, 600);
  cMean->Divide(2, 1);
  
  cMean->cd(1);
  gPad->SetLeftMargin(0.12);
  TMultiGraph* mgMean = new TMultiGraph();
  TLegend* legMean = new TLegend(0.55, 0.55, 0.88, 0.88);
  legMean->SetTextSize(0.028);
  
  for (size_t s = 0; s < samples.size(); ++s) {
    TGraph* g = new TGraph();
    int np = 0;
    for (const auto& pt : allResults[s]) g->SetPoint(np++, pt.p, pt.mean);
    g->SetMarkerStyle(samples[s].markerStyle);
    g->SetMarkerSize(0.5);
    g->SetMarkerColor(samples[s].color);
    g->SetLineColor(samples[s].color);
    g->SetLineWidth(samples[s].type == kReference ? 3 : (samples[s].type == kSimulation ? 2 : 1));
    g->SetLineStyle(samples[s].lineStyle);
    mgMean->Add(g, samples[s].type == kSimulation ? "LP" : "P");
    TString label = samples[s].label;
    if (samples[s].type == kReference) label += " [REF]";
    if (samples[s].type == kSimulation) label += " [SIM]";
    legMean->AddEntry(g, label, samples[s].type == kSimulation ? "lp" : "p");
  }
  
  mgMean->SetTitle(Form("%s: #Delta#beta Mean;p [MeV/c];#mu", particleName.Data()));
  mgMean->Draw("A");
  mgMean->GetYaxis()->SetRangeUser(-0.025, 0.025);
  TLine* zl1 = new TLine(0, 0, gMomentumMax, 0);
  zl1->SetLineStyle(kDashed); zl1->SetLineColor(kGray+1); zl1->Draw("same");
  legMean->Draw();
  
  cMean->cd(2);
  gPad->SetLeftMargin(0.12);
  TGraphAsymmErrors* gMeanBand = new TGraphAsymmErrors();
  TGraph* gMeanRef = new TGraph();
  int np = 0, npRef = 0;
  for (const auto& cp : comparison) {
    gMeanBand->SetPoint(np, cp.p, cp.meanAvg);
    gMeanBand->SetPointError(np++, 0, 0, cp.meanRMS, cp.meanRMS);
    if (cp.hasRef) gMeanRef->SetPoint(npRef++, cp.p, cp.meanRef);
  }
  gMeanBand->SetTitle(Form("%s: Exp Band #pm RMS;p [MeV/c];#mu", particleName.Data()));
  gMeanBand->SetFillColorAlpha(kBlue, 0.25);
  gMeanBand->SetLineColor(kBlue); gMeanBand->SetLineWidth(2);
  gMeanBand->Draw("A3");
  gMeanBand->GetYaxis()->SetRangeUser(-0.025, 0.025);
  
  if (npRef > 0) { gMeanRef->SetLineColor(kRed); gMeanRef->SetLineWidth(3); gMeanRef->Draw("L same"); }
  
  for (size_t s = 0; s < samples.size(); ++s) {
    if (samples[s].type != kSimulation) continue;
    TGraph* gSim = new TGraph();
    int nps = 0;
    for (const auto& pt : allResults[s]) gSim->SetPoint(nps++, pt.p, pt.mean);
    gSim->SetLineColor(samples[s].color); gSim->SetLineWidth(2); gSim->SetLineStyle(samples[s].lineStyle);
    gSim->Draw("L same");
  }
  
  TLine* zl2 = new TLine(0, 0, gMomentumMax, 0);
  zl2->SetLineStyle(kDashed); zl2->SetLineColor(kGray+1); zl2->Draw("same");
  
  TLegend* legBand = new TLegend(0.55, 0.70, 0.88, 0.88);
  legBand->AddEntry(gMeanBand, "Exp #pm RMS", "lf");
  if (npRef > 0) legBand->AddEntry(gMeanRef, "Reference", "l");
  for (size_t s = 0; s < samples.size(); ++s) {
    if (samples[s].type == kSimulation) {
      TGraph* d = new TGraph(); d->SetLineColor(samples[s].color); d->SetLineWidth(2); d->SetLineStyle(samples[s].lineStyle);
      legBand->AddEntry(d, samples[s].label + " [SIM]", "l");
    }
  }
  legBand->Draw();
  
  // ===== Plot 2: Sigma vs momentum =====
  TCanvas* cSigma = new TCanvas("c_sigma_comparison", "Sigma Comparison", 1400, 600);
  cSigma->Divide(2, 1);
  
  cSigma->cd(1);
  gPad->SetLeftMargin(0.12);
  TMultiGraph* mgSigma = new TMultiGraph();
  TLegend* legSigma = new TLegend(0.55, 0.55, 0.88, 0.88);
  legSigma->SetTextSize(0.028);
  
  for (size_t s = 0; s < samples.size(); ++s) {
    TGraph* g = new TGraph();
    int np = 0;
    for (const auto& pt : allResults[s]) g->SetPoint(np++, pt.p, pt.sigma);
    g->SetMarkerStyle(samples[s].markerStyle); g->SetMarkerSize(0.5);
    g->SetMarkerColor(samples[s].color); g->SetLineColor(samples[s].color);
    g->SetLineWidth(samples[s].type == kReference ? 3 : (samples[s].type == kSimulation ? 2 : 1));
    g->SetLineStyle(samples[s].lineStyle);
    mgSigma->Add(g, samples[s].type == kSimulation ? "LP" : "P");
    TString label = samples[s].label;
    if (samples[s].type == kReference) label += " [REF]";
    if (samples[s].type == kSimulation) label += " [SIM]";
    legSigma->AddEntry(g, label, samples[s].type == kSimulation ? "lp" : "p");
  }
  mgSigma->SetTitle(Form("%s: #Delta#beta Width;p [MeV/c];#sigma", particleName.Data()));
  mgSigma->Draw("A");
  legSigma->Draw();
  
  cSigma->cd(2);
  gPad->SetLeftMargin(0.12);
  TGraphAsymmErrors* gSigmaBand = new TGraphAsymmErrors();
  TGraph* gSigmaRef = new TGraph();
  np = 0; npRef = 0;
  for (const auto& cp : comparison) {
    gSigmaBand->SetPoint(np, cp.p, cp.sigmaAvg);
    gSigmaBand->SetPointError(np++, 0, 0, cp.sigmaRMS, cp.sigmaRMS);
    if (cp.hasRef) gSigmaRef->SetPoint(npRef++, cp.p, cp.sigmaRef);
  }
  gSigmaBand->SetTitle(Form("%s: Exp Band #pm RMS;p [MeV/c];#sigma", particleName.Data()));
  gSigmaBand->SetFillColorAlpha(kRed, 0.25);
  gSigmaBand->SetLineColor(kRed); gSigmaBand->SetLineWidth(2);
  gSigmaBand->Draw("A3");
  
  if (npRef > 0) { gSigmaRef->SetLineColor(kBlue); gSigmaRef->SetLineWidth(3); gSigmaRef->Draw("L same"); }
  
  for (size_t s = 0; s < samples.size(); ++s) {
    if (samples[s].type != kSimulation) continue;
    TGraph* gSim = new TGraph();
    int nps = 0;
    for (const auto& pt : allResults[s]) gSim->SetPoint(nps++, pt.p, pt.sigma);
    gSim->SetLineColor(samples[s].color); gSim->SetLineWidth(2); gSim->SetLineStyle(samples[s].lineStyle);
    gSim->Draw("L same");
  }
  
  TLegend* legSBand = new TLegend(0.55, 0.70, 0.88, 0.88);
  legSBand->AddEntry(gSigmaBand, "Exp #pm RMS", "lf");
  if (npRef > 0) legSBand->AddEntry(gSigmaRef, "Reference", "l");
  for (size_t s = 0; s < samples.size(); ++s) {
    if (samples[s].type == kSimulation) {
      TGraph* d = new TGraph(); d->SetLineColor(samples[s].color); d->SetLineWidth(2); d->SetLineStyle(samples[s].lineStyle);
      legSBand->AddEntry(d, samples[s].label + " [SIM]", "l");
    }
  }
  legSBand->Draw();
  
  // ===== Plot 3: Data vs Simulation =====
  bool hasSim = false;
  for (const auto& s : samples) if (s.type == kSimulation) hasSim = true;
  
  if (hasSim) {
    TCanvas* cDS = new TCanvas("c_data_sim", "Data vs Simulation", 1200, 600);
    cDS->Divide(2, 1);
    
    cDS->cd(1);
    gPad->SetLeftMargin(0.14); gPad->SetGridy();
    TMultiGraph* mgDM = new TMultiGraph();
    for (size_t s = 0; s < samples.size(); ++s) {
      if (samples[s].type != kSimulation) continue;
      TGraph* gD = new TGraph();
      int npts = 0;
      for (const auto& cp : comparison) {
        for (size_t i = 0; i < cp.simLabels.size(); ++i) {
          if (cp.simLabels[i] == samples[s].label) {
            gD->SetPoint(npts++, cp.p, cp.meanAvg - cp.meanSim[i]);
            break;
          }
        }
      }
      gD->SetLineColor(samples[s].color); gD->SetLineWidth(2);
      gD->SetMarkerColor(samples[s].color); gD->SetMarkerStyle(20); gD->SetMarkerSize(0.4);
      mgDM->Add(gD, "LP");
    }
    mgDM->SetTitle("#mu: Data - Simulation;p [MeV/c];#Delta#mu");
    mgDM->Draw("A");
    TLine* zd1 = new TLine(0, 0, gMomentumMax, 0);
    zd1->SetLineStyle(kDashed); zd1->SetLineColor(kGray+1); zd1->Draw("same");
    
    cDS->cd(2);
    gPad->SetLeftMargin(0.14); gPad->SetGridy();
    TMultiGraph* mgDS = new TMultiGraph();
    for (size_t s = 0; s < samples.size(); ++s) {
      if (samples[s].type != kSimulation) continue;
      TGraph* gD = new TGraph();
      int npts = 0;
      for (const auto& cp : comparison) {
        for (size_t i = 0; i < cp.simLabels.size(); ++i) {
          if (cp.simLabels[i] == samples[s].label) {
            gD->SetPoint(npts++, cp.p, cp.sigmaAvg - cp.sigmaSim[i]);
            break;
          }
        }
      }
      gD->SetLineColor(samples[s].color); gD->SetLineWidth(2);
      gD->SetMarkerColor(samples[s].color); gD->SetMarkerStyle(20); gD->SetMarkerSize(0.4);
      mgDS->Add(gD, "LP");
    }
    mgDS->SetTitle("#sigma: Data - Simulation;p [MeV/c];#Delta#sigma");
    mgDS->Draw("A");
    TLine* zd2 = new TLine(0, 0, gMomentumMax, 0);
    zd2->SetLineStyle(kDashed); zd2->SetLineColor(kGray+1); zd2->Draw("same");
  }
  
  // ===== Plot 4: Relative uncertainty =====
  TCanvas* cU = new TCanvas("c_uncertainty", "Relative Uncertainty", 800, 600);
  gPad->SetLeftMargin(0.12); gPad->SetGridy();
  TGraph* gRel = new TGraph();
  for (size_t i = 0; i < comparison.size(); ++i) {
    double rel = (comparison[i].sigmaAvg != 0) ? comparison[i].sigmaRMS / comparison[i].sigmaAvg : 0;
    gRel->SetPoint(i, comparison[i].p, 100 * std::min(rel, 1.0));
  }
  gRel->SetLineColor(kRed); gRel->SetLineWidth(2); gRel->SetFillColorAlpha(kRed, 0.2);
  gRel->SetTitle(Form("%s: Relative Systematic (Exp only);p [MeV/c];#sigma Uncertainty [%%]", particleName.Data()));
  gRel->Draw("ALF");
  gRel->GetYaxis()->SetRangeUser(0, 30);
  TLatex tx; tx.SetNDC(); tx.SetTextSize(0.025); tx.SetTextColor(kGray+2);
  tx.DrawLatex(0.15, 0.85, "Based on experimental samples only");
}

// =====================================================
// SAVE RESULTS - Updated for 5 sigma levels
// =====================================================
void saveResults(const TString& outputFile, const TString& particleName,
                 const vector<SampleInfo>& samples, const vector<ComparisonPoint>& comparison,
                 TCutG* env1, TCutG* env2p5, TCutG* env3, TCutG* env3p5, TCutG* env5, TCutG* ref3) {
  TFile* fOut = new TFile(outputFile, "RECREATE");
  
  TTree* tree = new TTree("SystematicComparison", "PID systematic comparison");
  Double_t t_p, t_meanAvg, t_meanRMS, t_sigmaAvg, t_sigmaRMS, t_meanRef, t_sigmaRef;
  Int_t t_nSamples, t_hasRef;
  tree->Branch("p", &t_p); tree->Branch("mean_avg", &t_meanAvg); tree->Branch("mean_rms", &t_meanRMS);
  tree->Branch("sigma_avg", &t_sigmaAvg); tree->Branch("sigma_rms", &t_sigmaRMS);
  tree->Branch("mean_ref", &t_meanRef); tree->Branch("sigma_ref", &t_sigmaRef);
  tree->Branch("n_samples", &t_nSamples); tree->Branch("has_ref", &t_hasRef);
  
  for (const auto& cp : comparison) {
    t_p = cp.p; t_meanAvg = cp.meanAvg; t_meanRMS = cp.meanRMS;
    t_sigmaAvg = cp.sigmaAvg; t_sigmaRMS = cp.sigmaRMS;
    t_meanRef = cp.meanRef; t_sigmaRef = cp.sigmaRef;
    t_nSamples = cp.nExpSamples; t_hasRef = cp.hasRef ? 1 : 0;
    tree->Fill();
  }
  tree->Write();
  
  // Save all 5 sigma levels: 1, 2.5, 3, 3.5, 5
  TDirectory* dE = fOut->mkdir("EnvelopeCuts"); dE->cd();
  if (env1) env1->Write();
  if (env2p5) env2p5->Write();
  if (env3) env3->Write();
  if (env3p5) env3p5->Write();
  if (env5) env5->Write();
  
  TDirectory* dR = fOut->mkdir("ReferenceCut"); dR->cd();
  if (ref3) ref3->Write();
  
  fOut->Close(); delete fOut;
  cout << "[INFO] Saved to " << outputFile << endl;
  cout << "  Envelope cuts saved: 1σ, 2.5σ, 3σ, 3.5σ, 5σ" << endl;
}

// =====================================================
// PRINT SUMMARY
// =====================================================
void printSummaryTable(const vector<SampleInfo>& samples, const vector<ComparisonPoint>& comparison,
                       const TString& particleName) {
  cout << "\n" << string(70, '=') << "\n  SYSTEMATIC SUMMARY: " << particleName << " (π+)" << "\n" << string(70, '=') << endl;
  
  cout << "\nEXPERIMENTAL samples (in systematics):" << endl;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].type == kSimulation) continue;
    cout << Form("  [%zu] %-15s %s", i, samples[i].label.Data(), 
                 samples[i].type == kReference ? "<-- REFERENCE" : "") << endl;
  }
  
  cout << "\nSIMULATION samples (comparison only, NOT in systematics):" << endl;
  bool hasSim = false;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].type != kSimulation) continue;
    hasSim = true;
    cout << Form("  [%zu] %-15s", i, samples[i].label.Data()) << endl;
  }
  if (!hasSim) cout << "  (none)" << endl;
  
  double totalSigmaRel = 0; int n = 0;
  for (const auto& cp : comparison) {
    if (cp.sigmaAvg != 0) { totalSigmaRel += cp.sigmaRMS / cp.sigmaAvg; n++; }
  }
  
  cout << "\n" << string(70, '-') << endl;
  cout << Form("OVERALL σ relative uncertainty: %.1f%%", n > 0 ? 100*totalSigmaRel/n : 0) << endl;
  cout << string(70, '=') << endl;
}

// =====================================================
// MAIN FUNCTION - PION+ (pip)
// =====================================================
void pid_systematics_calculator_pip(
    const TString& particleName = "pip",
    const TString& outputFile = "pid_pip_systematics.root",
    int widthIdx = 1
) {
  cout << "\n" << string(60, '=') << "\n  PID SYSTEMATICS CALCULATOR (π+)" << "\n" << string(60, '=') << endl;
  
  // =====================================================
  // CONFIGURE YOUR INPUT FILES HERE - PION+ (pip)
  // =====================================================
  vector<SampleInfo> samples;
  
  // EXPERIMENTAL (included in systematics)
  samples.push_back({"pip_pid_cuts_049_exp.root", "Run 049", "temporal", kReference, kRed, 20, 1});
  samples.push_back({"pip_pid_cuts_049_exp.root", "Run 049 copy", "temporal", kReference, kRed, 21, 1});
  //samples.push_back({"pip_pid_cuts_050_exp.root", "Run 050", "temporal", kExperimental, kBlue, 21, 1});
  //samples.push_back({"pip_pid_cuts_052_exp.root", "Run 051", "temporal", kExperimental, kGreen+2, 22, 1});
  //samples.push_back({"pip_pid_cuts_060_exp.root", "Run 060", "temporal", kExperimental, kMagenta, 23, 1});
  //samples.push_back({"pip_pid_cuts_066_exp.root", "Run 066", "temporal", kExperimental, kYellow+1, 24, 1});
  
  // SIMULATION (comparison only - NOT in systematics!)
  samples.push_back({"pip_pid_cuts_pp45_sim.root", "SMASH", "simulation", kSimulation, kBlack, 25, 2});
  
  // Check files
  vector<SampleInfo> valid;
  int nExp = 0, nSim = 0;
  bool hasRef = false;
  for (auto& s : samples) {
    if (!gSystem->AccessPathName(s.filename)) {
      valid.push_back(s);
      if (s.type == kSimulation) nSim++; else nExp++;
      if (s.type == kReference) hasRef = true;
    } else {
      cout << "[WARN] Not found: " << s.filename << endl;
    }
  }
  
  cout << "Found " << nExp << " experimental + " << nSim << " simulation files." << endl;
  if (nExp < 1) { cerr << "[ERROR] Need at least 1 experimental file!" << endl; return; }
  
  if (!hasRef) {
    for (auto& s : valid) if (s.type == kExperimental) { s.type = kReference; break; }
  }
  
  // Load data
  vector<vector<FitPoint>> allResults;
  for (const auto& s : valid) {
    bool isSim = (s.type == kSimulation);
    allResults.push_back(loadFitResults(s.filename, widthIdx, false, isSim));
  }
  
  // Compare
  vector<ComparisonPoint> comparison = compareFitResults(valid, allResults, 10.0);
  if (comparison.empty()) { cerr << "[ERROR] No comparison points!" << endl; return; }
  
  // Envelope cuts (experimental only) - Pion mass
  // Now loading 5 sigma levels: 1, 2.5, 3, 3.5, 5
  double mass = 139.57;
  vector<TCutG*> c1, c2p5, c3, c3p5, c5;
  TCutG* ref3 = nullptr;
  
  for (size_t s = 0; s < valid.size(); ++s) {
    if (valid[s].type == kSimulation) continue;
    
    // Cut names for all 5 sigma levels
    TString n1 = Form("cut_%s_1sig_w%d", particleName.Data(), widthIdx);
    TString n2p5 = Form("cut_%s_25sig_w%d", particleName.Data(), widthIdx);
    TString n3 = Form("cut_%s_3sig_w%d", particleName.Data(), widthIdx);
    TString n3p5 = Form("cut_%s_35sig_w%d", particleName.Data(), widthIdx);
    TString n5 = Form("cut_%s_5sig_w%d", particleName.Data(), widthIdx);
    
    // Load all sigma levels
    TCutG* t1 = loadTCutG(valid[s].filename, n1); if (t1) c1.push_back(t1);
    TCutG* t2p5 = loadTCutG(valid[s].filename, n2p5); if (t2p5) c2p5.push_back(t2p5);
    TCutG* t3 = loadTCutG(valid[s].filename, n3); if (t3) c3.push_back(t3);
    TCutG* t3p5 = loadTCutG(valid[s].filename, n3p5); if (t3p5) c3p5.push_back(t3p5);
    TCutG* t5 = loadTCutG(valid[s].filename, n5); if (t5) c5.push_back(t5);
    
    if (valid[s].type == kReference && t3) ref3 = (TCutG*)t3->Clone(Form("ref_%s_3sig", particleName.Data()));
  }
  
  // Generate envelope cuts for all 5 sigma levels
  TCutG* env1 = generateEnvelopeCut(c1, Form("envelope_%s_1sig", particleName.Data()), mass, 0, gMomentumMax);
  TCutG* env2p5 = generateEnvelopeCut(c2p5, Form("envelope_%s_25sig", particleName.Data()), mass, 0, gMomentumMax);
  TCutG* env3 = generateEnvelopeCut(c3, Form("envelope_%s_3sig", particleName.Data()), mass, 0, gMomentumMax);
  TCutG* env3p5 = generateEnvelopeCut(c3p5, Form("envelope_%s_35sig", particleName.Data()), mass, 0, gMomentumMax);
  TCutG* env5 = generateEnvelopeCut(c5, Form("envelope_%s_5sig", particleName.Data()), mass, 0, gMomentumMax);
  
  // Report which envelope cuts were successfully created
  cout << "\n[INFO] Envelope cuts generated:" << endl;
  cout << "  1σ:   " << (env1 ? "OK" : "FAILED (no input cuts)") << endl;
  cout << "  2.5σ: " << (env2p5 ? "OK" : "FAILED (no input cuts)") << endl;
  cout << "  3σ:   " << (env3 ? "OK" : "FAILED (no input cuts)") << endl;
  cout << "  3.5σ: " << (env3p5 ? "OK" : "FAILED (no input cuts)") << endl;
  cout << "  5σ:   " << (env5 ? "OK" : "FAILED (no input cuts)") << endl;
  
  // Plots
  TString prefix = outputFile; prefix.ReplaceAll(".root", "");
  createComparisonPlots(valid, allResults, comparison, particleName, prefix);
  
  // Save - now includes all 5 sigma levels
  saveResults(outputFile, particleName, valid, comparison, env1, env2p5, env3, env3p5, env5, ref3);
  printSummaryTable(valid, comparison, particleName);
  
  // Calculate yield systematics
  cout << "\n\n";
  cout << string(80, '#') << endl;
  cout << "#  DETAILED SYSTEMATIC ANALYSIS (π+)" << endl;
  cout << string(80, '#') << endl;
  
  calculateYieldSystematics(comparison, 3.0);
  calculateYieldSystematics(comparison, 5.0);
  
  // Calculate simulation mismatch
  calculateSimMismatch(comparison, 3.0);
  calculateSimMismatch(comparison, 5.0);
  
  // Create TCutG comparison plot (without zoomed version)
  TString refFile = "", simFile = "";
  for (const auto& s : valid) {
    if (s.type == kReference) refFile = s.filename;
    if (s.type == kSimulation) simFile = s.filename;
  }
  if (!refFile.IsNull() && !simFile.IsNull()) {
    plotTCutGComparison(refFile, simFile, particleName, prefix, widthIdx);
  }
}
