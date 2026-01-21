// pid_macro_tcutg_p_sim.C
// PROTON PID analysis using Δβ = β_measured - β_proton(p) representation
// WITH TCutG GENERATION
// 
// ROBUST FITTING STRATEGY:
// 1. Find peak maximum and 80% boundaries in data
// 2. Preliminary Gaussian fit to peak top - ANCHORS signal position
// 3. Full fit: Signal Gauss + Background Gauss + Polynomial
// 4. SUBTRACT polynomial and background Gauss from data
// 5. Final Gaussian fit to cleaned data - USE THESE for contours
//
// Scanning strategy:
// - Warmup: [310, 350] MeV/c
// - Anchor: 310 MeV/c
// - Phase 0: right edge at 310, width doubling going left to 0
// - Phase 1: forward from 310 to 900 with 1 MeV/c steps
// - Phase 2: above 900, progressive doubling with sliding:
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

// =====================================================
// ADDITION #1: Additional includes for TCutG generation
// =====================================================
#include "TSpline.h"
#include "TCutG.h"
#include "TNamed.h"
// =====================================================

using std::cout; using std::endl;

// =====================================================
// GLOBAL CONSTANTS
// =====================================================
const double gProtonMass = 938.272;  // MeV/c²
const double gSSquared = 1e-4;       // For (mass, a) transformation

// =====================================================
// PHYSICS FUNCTIONS
// =====================================================

double betaProton(double p) {
  return p / std::sqrt(p * p + gProtonMass * gProtonMass);
}

double deltaBetaToBeta(double p, double dBeta) {
  return betaProton(p) + dBeta;
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
  double mean;        // Δβ mean (should be near 0 for protons)
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
// ADDITION #2: ContourOutput structure for TCutG generation
// =====================================================
struct ContourOutput {
  TGraph* meanGraph;       // μ(p) after smoothing - in Δβ space
  TGraph* sigmaGraph;      // σ(p) after smoothing - in Δβ space
  TSpline3* meanSpline;    // Interpolated μ(p) for smooth evaluation
  TSpline3* sigmaSpline;   // Interpolated σ(p) for smooth evaluation
  TCutG* cut1sig;          // 1σ contour in (p, β) plane
  TCutG* cut25sig;         // 2.5σ contour in (p, β) plane
  TCutG* cut3sig;          // 3σ contour in (p, β) plane
  TCutG* cut35sig;         // 3.5σ contour in (p, β) plane
  TCutG* cut5sig;          // 5σ contour in (p, β) plane
  int selectedWidth;       // Which width was used (0=1x, 1=2x, 2=4x, 3=8x)
  bool valid;              // Whether generation succeeded
};
// =====================================================

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
// ADDITION #3: TCutG Generation Function
// =====================================================
ContourOutput generateSmoothedCuts(
    const std::vector<FitResult>& fitResults,
    const std::vector<double>& momCenters,
    int selectedWidth,
    double particleMass,
    double pMin, 
    double pMax,
    double pStep,
    const TString& particleName,
    const TString& varNameX,
    const TString& varNameY,
    int medianWindow,
    int gaussWindow
) {
  ContourOutput out;
  out.valid = false;
  out.selectedWidth = selectedWidth;
  out.meanGraph = nullptr;
  out.sigmaGraph = nullptr;
  out.meanSpline = nullptr;
  out.sigmaSpline = nullptr;
  out.cut1sig = nullptr;
  out.cut25sig = nullptr;
  out.cut3sig = nullptr;
  out.cut35sig = nullptr;
  out.cut5sig = nullptr;

  // --- AVERAGING approach: bin by momentum, average all mu/sigma values ---
  // For Phase 2 (high momentum), we extend using the last good values
  
  double maxBinWidth = 80.0;  // Only use fits from narrow bins (Phase 1 quality)
  
  // Debug counters
  int nTotal = 0, nNotSuccess = 0, nBadChi2 = 0, nBadSigma = 0, nBadMean = 0, nTooWide = 0;
  
  // First pass: collect all valid results
  struct ValidPoint {
    double p, mean, sigma;
  };
  std::vector<ValidPoint> validPoints;
  
  for (size_t i = 0; i < fitResults.size(); ++i) {
    const FitResult& r = fitResults[i];
    nTotal++;
    
    if (!r.success) { nNotSuccess++; continue; }
    if (r.chi2ndf > 50.0) { nBadChi2++; continue; }
    if (r.sigma < 0.002 || r.sigma > 0.10) { nBadSigma++; continue; }
    if (std::abs(r.mean) > 0.08) { nBadMean++; continue; }
    
    // FILTER: only use fits from narrow bins (local fits)
    double binWidth = r.pHigh - r.pLow;
    if (binWidth > maxBinWidth) { nTooWide++; continue; }
    
    ValidPoint vp;
    vp.p = momCenters[i];
    vp.mean = r.mean;
    vp.sigma = r.sigma;
    validPoints.push_back(vp);
  }
  
  cout << "[TCutG] Quality cut summary: total=" << nTotal 
       << ", not_success=" << nNotSuccess
       << ", bad_chi2=" << nBadChi2
       << ", bad_sigma=" << nBadSigma
       << ", bad_mean=" << nBadMean
       << ", too_wide=" << nTooWide
       << ", passed=" << validPoints.size() << endl;
  
  cout << "[TCutG] Selected width " << selectedWidth 
       << ": " << validPoints.size() << " valid points for contour generation" << endl;
  
  if (validPoints.size() < 10) {
    cerr << "[TCutG] ERROR: Too few valid points (<10) for spline fitting!" << endl;
    return out;
  }
  
  // Second pass: bin by momentum and average
  double binSize = 20.0;  // 20 MeV/c bins for averaging
  double pMinData = 1e9, pMaxData = -1e9;
  for (const auto& vp : validPoints) {
    if (vp.p < pMinData) pMinData = vp.p;
    if (vp.p > pMaxData) pMaxData = vp.p;
  }
  
  std::vector<double> rawP, rawMean, rawSigma;
  
  for (double pBin = pMinData; pBin <= pMaxData; pBin += binSize) {
    double sumMean = 0, sumSigma = 0;
    int count = 0;
    
    for (const auto& vp : validPoints) {
      if (vp.p >= pBin && vp.p < pBin + binSize) {
        sumMean += vp.mean;
        sumSigma += vp.sigma;
        count++;
      }
    }
    
    if (count > 0) {
      rawP.push_back(pBin + binSize/2);  // Bin center
      rawMean.push_back(sumMean / count);
      rawSigma.push_back(sumSigma / count);
    }
  }
  
  cout << "[TCutG] Averaged bins from good fits: " << rawP.size() 
       << " points, range [" << rawP.front() << ", " << rawP.back() << "] MeV/c" << endl;
  
  // === EXTENSION TO LOW MOMENTUM ===
  double extendDownTo = 100.0;  // Extend down to 100 MeV/c
  
  if (rawP.front() > extendDownTo) {
    // Get average of first few points for stable extension
    int nAvg = std::min(5, (int)rawP.size());
    double firstMean = 0, firstSigma = 0;
    for (int i = 0; i < nAvg; ++i) {
      firstMean += rawMean[i];
      firstSigma += rawSigma[i];
    }
    firstMean /= nAvg;
    firstSigma /= nAvg;
    
    cout << "[TCutG] Extending down from " << rawP.front() << " to " << extendDownTo 
         << " MeV/c with mu=" << firstMean << ", sigma=" << firstSigma << endl;
    
    // Add extension points at the beginning (insert in reverse order)
    std::vector<double> lowP, lowMean, lowSigma;
    for (double p = rawP.front() - binSize; p >= extendDownTo; p -= binSize) {
      lowP.push_back(p);
      lowMean.push_back(firstMean);
      lowSigma.push_back(firstSigma);
    }
    
    // Reverse and prepend
    std::reverse(lowP.begin(), lowP.end());
    std::reverse(lowMean.begin(), lowMean.end());
    std::reverse(lowSigma.begin(), lowSigma.end());
    
    lowP.insert(lowP.end(), rawP.begin(), rawP.end());
    lowMean.insert(lowMean.end(), rawMean.begin(), rawMean.end());
    lowSigma.insert(lowSigma.end(), rawSigma.begin(), rawSigma.end());
    
    rawP = lowP;
    rawMean = lowMean;
    rawSigma = lowSigma;
  }
  
  // === EXTENSION TO HIGH MOMENTUM ===
  // At high p, beta resolution is dominated by detector, so sigma ~constant
  double extendTo = 4500.0;  // Extend to 4500 MeV/c
  
  if (rawP.back() < extendTo) {
    // Get average of last few points for stable extension
    int nAvg = std::min(5, (int)rawP.size());
    double lastMean = 0, lastSigma = 0;
    for (int i = rawP.size() - nAvg; i < (int)rawP.size(); ++i) {
      lastMean += rawMean[i];
      lastSigma += rawSigma[i];
    }
    lastMean /= nAvg;
    lastSigma /= nAvg;
    
    cout << "[TCutG] Extending from " << rawP.back() << " to " << extendTo 
         << " MeV/c with mu=" << lastMean << ", sigma=" << lastSigma << endl;
    
    // Add extension points
    for (double p = rawP.back() + binSize; p <= extendTo; p += binSize) {
      rawP.push_back(p);
      rawMean.push_back(lastMean);
      rawSigma.push_back(lastSigma);
    }
  }
  
  cout << "[TCutG] After extension: " << rawP.size() << " points" << endl;
  
  if (rawP.size() < 10) {
    cerr << "[TCutG] ERROR: Too few averaged points (<10) for spline fitting!" << endl;
    return out;
  }
  
  // --- NO smoothing, NO splines - use averaged values directly ---
  // The averaging already smooths out the Phase 2 chaos
  
  // --- Create TGraphs (for storage/reference only) ---
  int nPts = rawP.size();
  out.meanGraph = new TGraph(nPts, rawP.data(), rawMean.data());
  out.sigmaGraph = new TGraph(nPts, rawP.data(), rawSigma.data());
  
  out.meanGraph->SetName(Form("g_%s_mean_w%d", particleName.Data(), selectedWidth));
  out.meanGraph->SetTitle(Form("%s #Delta#beta mean vs p (width %d)", particleName.Data(), selectedWidth));
  out.sigmaGraph->SetName(Form("g_%s_sigma_w%d", particleName.Data(), selectedWidth));
  out.sigmaGraph->SetTitle(Form("%s #Delta#beta sigma vs p (width %d)", particleName.Data(), selectedWidth));
  
  // Create dummy splines for compatibility (but we won't use them)
  out.meanSpline = new TSpline3("dummy_mean", out.meanGraph);
  out.sigmaSpline = new TSpline3("dummy_sigma", out.sigmaGraph);
  
  cout << "[TCutG] Momentum range: [" << rawP.front() << ", " << rawP.back() << "] MeV/c" << endl;
  
  // --- BUILD TCutG DIRECTLY from averaged points - NO SPLINE INTERPOLATION ---
  auto makeCut = [&](double nSigma, const char* cutName) -> TCutG* {
    std::vector<double> pCut, betaCut;
    
    // Forward pass: UPPER boundary (use averaged values directly)
    for (size_t i = 0; i < rawP.size(); ++i) {
      double p = rawP[i];
      double mu = rawMean[i];
      double sig = rawSigma[i];
      double betaTheory = p / std::sqrt(p*p + particleMass*particleMass);
      double betaUpper = betaTheory + mu + nSigma * sig;
      
      if (betaUpper > 0.01 && betaUpper < 1.5) {
        pCut.push_back(p);
        betaCut.push_back(betaUpper);
      }
    }
    
    // Backward pass: LOWER boundary (closes the polygon)
    for (int i = rawP.size() - 1; i >= 0; --i) {
      double p = rawP[i];
      double mu = rawMean[i];
      double sig = rawSigma[i];
      double betaTheory = p / std::sqrt(p*p + particleMass*particleMass);
      double betaLower = betaTheory + mu - nSigma * sig;
      
      if (betaLower > 0.01 && betaLower < 1.5) {
        pCut.push_back(p);
        betaCut.push_back(betaLower);
      }
    }
    
    // Close the polygon
    if (!pCut.empty()) {
      pCut.push_back(pCut[0]);
      betaCut.push_back(betaCut[0]);
    }
    
    TCutG* cut = new TCutG(cutName, pCut.size(), pCut.data(), betaCut.data());
    cut->SetVarX(varNameX.Data());
    cut->SetVarY(varNameY.Data());
    
    return cut;
  };
  
  // --- Generate cuts for 1σ, 2.5σ, 3σ, 3.5σ, 5σ ---
  out.cut1sig = makeCut(1.0, Form("cut_%s_1sig_w%d", particleName.Data(), selectedWidth));
  out.cut25sig = makeCut(2.5, Form("cut_%s_25sig_w%d", particleName.Data(), selectedWidth));
  out.cut3sig = makeCut(3.0, Form("cut_%s_3sig_w%d", particleName.Data(), selectedWidth));
  out.cut35sig = makeCut(3.5, Form("cut_%s_35sig_w%d", particleName.Data(), selectedWidth));
  out.cut5sig = makeCut(5.0, Form("cut_%s_5sig_w%d", particleName.Data(), selectedWidth));
  
  // --- Set visual properties ---
  out.cut1sig->SetLineColor(kRed);
  out.cut1sig->SetLineWidth(2);
  out.cut1sig->SetLineStyle(kSolid);
  
  out.cut3sig->SetLineColor(kBlue);
  out.cut3sig->SetLineWidth(2);
  out.cut3sig->SetLineStyle(kDashed);
  
  out.cut5sig->SetLineColor(kGreen+2);
  out.cut5sig->SetLineWidth(2);
  out.cut5sig->SetLineStyle(7);
  
  out.valid = true;
  
  cout << "[TCutG] Successfully generated cuts: " 
       << out.cut1sig->GetName() << ", "
       << out.cut3sig->GetName() << ", "
       << out.cut5sig->GetName() << endl;
  
  return out;
}

// =====================================================
// ADDITION #4: Save Results Function
// =====================================================
void saveContourResults(
    const ContourOutput& contours,
    const std::vector<std::vector<FitResult>>& allFitResults,
    const std::vector<std::vector<double>>& allMomCenters,
    const TString widthLabels[],
    int nWidths,
    const TString& outputFileName,
    const TString& particleName
) {
  TFile* fOut = new TFile(outputFileName, "RECREATE");
  
  if (!fOut || fOut->IsZombie()) {
    cerr << "[TCutG] ERROR: Cannot create output file " << outputFileName << endl;
    return;
  }
  
  cout << "\n[TCutG] Saving results to " << outputFileName << endl;
  
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
  
  if (contours.valid) {
    TDirectory* dirContours = fOut->mkdir("Contours");
    dirContours->cd();
    
    contours.meanGraph->Write();
    contours.sigmaGraph->Write();
    contours.meanSpline->Write();
    contours.sigmaSpline->Write();
    
    cout << "  - Smoothed TGraphs and TSpline3 objects" << endl;
    
    TDirectory* dirCuts = fOut->mkdir("TCutG");
    dirCuts->cd();
    
    contours.cut1sig->Write();
    contours.cut25sig->Write();
    contours.cut3sig->Write();
    contours.cut35sig->Write();
    contours.cut5sig->Write();
    
    cout << "  - TCutG objects: " << contours.cut1sig->GetName() << ", "
         << contours.cut25sig->GetName() << ", "
         << contours.cut3sig->GetName() << ", " 
         << contours.cut35sig->GetName() << ", "
         << contours.cut5sig->GetName() << endl;
  }
  
  fOut->cd();
  TNamed* metaParticle = new TNamed("particle", particleName.Data());
  TNamed* metaWidth = new TNamed("selected_width", Form("%d", contours.selectedWidth));
  TNamed* metaWidthLabel = new TNamed("width_label", widthLabels[contours.selectedWidth].Data());
  metaParticle->Write();
  metaWidth->Write();
  metaWidthLabel->Write();
  
  fOut->Close();
  delete fOut;
  
  cout << "[TCutG] Output file saved successfully.\n" << endl;
}

// =====================================================
// ADDITION #5: Draw Cuts on Histogram Function
// =====================================================
TCanvas* drawCutsOnHistogram(
    TH2F* h2PB,
    const ContourOutput& contours,
    const TString& canvasName,
    double particleMass
) {
  if (!contours.valid || !h2PB) {
    cerr << "[TCutG] Cannot draw: invalid contours or histogram" << endl;
    return nullptr;
  }
  
  TCanvas* c = new TCanvas(canvasName, "PID Cuts in (p, #beta)", 1000, 800);
  c->cd();
  
  h2PB->Draw("colz");
  
  TF1* theoryCurve = new TF1("theoryCurve", 
    Form("x/sqrt(x*x + %f*%f)", particleMass, particleMass), 0, 4500);
  theoryCurve->SetLineColor(kBlack);
  theoryCurve->SetLineStyle(kDashed);
  theoryCurve->SetLineWidth(2);
  theoryCurve->Draw("same");
  
  contours.cut1sig->Draw("L same");
  contours.cut3sig->Draw("L same");
  contours.cut5sig->Draw("L same");
  
  TLegend* leg = new TLegend(0.55, 0.15, 0.88, 0.40);
  leg->SetHeader(Form("Width %d cuts", contours.selectedWidth));
  leg->AddEntry(contours.cut1sig, "1#sigma", "l");
  leg->AddEntry(contours.cut3sig, "3#sigma", "l");
  leg->AddEntry(contours.cut5sig, "5#sigma", "l");
  leg->AddEntry(theoryCurve, "Theory", "l");
  leg->Draw();
  
  c->Modified();
  c->Update();
  
  return c;
}
// =====================================================
// END OF TCutG ADDITIONS
// =====================================================


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
    if (chi2ndf > 300.0) continue;
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

void pid_macro_tcutg_p_sim_extended() {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(111);

  // --- Build TChain
  const char* treeName = "P";
  const std::vector<TString> files = {
	  #include "SMASH/smash_100.list"
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
  
  // Main histogram: p vs Δβ (proton)
  const char* h2name = "h2_p_deltaBeta";
  TString drawCmd = Form(
    "p_beta - p_p/sqrt(p_p*p_p + %.2f*%.2f) : p_p >> %s(450,0,4500,300,-0.15,0.15)",
    gProtonMass, gProtonMass, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "isBest==1 && eVertReco_z>-500 && p_sim_id==14", "colz");
  TH2F* h2DB = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2DB) {
    cout << "Failed to create histogram" << endl;
    return;
  }

  h2DB->SetTitle("Momentum vs #Delta#beta (#beta - #beta_{p})");
  h2DB->GetXaxis()->SetTitle("Momentum [MeV/c]");
  h2DB->GetYaxis()->SetTitle("#Delta#beta = #beta - #beta_{p}");

  // (p, β) histogram
  const char* h2pb_name = "h2_p_beta";
  if (gDirectory->FindObject(h2pb_name)) gDirectory->Delete(Form("%s;*", h2pb_name));
  chain->Draw(Form("p_beta : p_p >> %s(450,0,4500,300,0.1,1.15)", h2pb_name), "isBest==1 && eVertReco_z>-500 && p_sim_id==14", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2pb_name));

  // (mass, a) histogram
  const char* h2ma_name = "h2_mass_a";
  TString drawMA = Form(
    "sqrt(1 + %.1e*p_p*p_p - pow(1-p_beta*p_beta,2)) : "
    "p_p*sqrt(1/pow(p_beta,2) - 1) >> %s(400,0,2000,160,0,16)",
    gSSquared, h2ma_name);
  if (gDirectory->FindObject(h2ma_name)) gDirectory->Delete(Form("%s;*", h2ma_name));
  chain->Draw(drawMA, "isBest==1 && eVertReco_z>-500 && p_sim_id==14 && p_beta>0 && p_beta<1", "colz");
  TH2F* h2MA = static_cast<TH2F*>(gDirectory->Get(h2ma_name));

  // (p, mass²) histogram - mass² = p² * (1/β² - 1)
  const char* h2m2_name = "h2_p_mass2";
  if (gDirectory->FindObject(h2m2_name)) gDirectory->Delete(Form("%s;*", h2m2_name));
  chain->Draw(Form("p_p*p_p*(1.0/(p_beta*p_beta) - 1) : p_p >> %s(450,0,4500,400,0,2000000)", h2m2_name), 
              "isBest==1 && eVertReco_z>-500 && p_sim_id==14 && p_beta>0.1 && p_beta<1.5", "colz");
  TH2F* h2M2 = static_cast<TH2F*>(gDirectory->Get(h2m2_name));
  if (h2M2) {
    h2M2->SetTitle("Mass^{2} vs Momentum (Proton)");
    h2M2->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2M2->GetYaxis()->SetTitle("Mass^{2} [MeV^{2}/c^{4}]");
  }

  // =====================================================
  // SCANNING PARAMETERS - PROTON
  // =====================================================
  
  const double warmupLow = 310.0;       // Warmup fit range [310, 350]
  const double warmupHigh = 350.0;
  const double startMom = 310.0;        // Anchor point
  const double transitionMom = 900.0;   // Where Phase 2 doubling starts
  const double endMom = 4500.0;         // Extended range for protons
  const double stepSize = 1.0;          // 1 MeV/c steps in Phase 1
  
  const int nWidths = 4;
  const double baseWidths[nWidths] = {10.0, 20.0, 40.0, 80.0};  // Larger widths for protons
  const TString widthLabels[nWidths] = {"1x(10)", "2x(20)", "4x(40)", "8x(80)"};
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
    // Phase 0: RIGHT EDGE anchored at startMom (310), width DOUBLES going left
    // Phase 1: forward from startMom to transitionMom with 1 MeV/c steps
    // Phase 2: forward from transitionMom with doubling width
    // =====================================================
    
    std::vector<double> momLows, momHighs, momCenters, sliceWidths;
    std::vector<int> phaseTag;

    // Phase 0: RIGHT EDGE anchored at startMom (310), width DOUBLES going left
    // For baseW=10: [300,310], [280,310], [250,310], [190,310], [0,310]
    // Width: 10 → 20 → 40 → 80 → ... until left reaches 0
    std::vector<int> phase0Indices;
    {
      double pRight = startMom;  // Always 310
      double currentWidth = baseW;
      double pLeft = pRight - currentWidth;
      
      while (pLeft > 0) {
        int idx = momLows.size();
        momLows.push_back(pLeft);
        momHighs.push_back(pRight);  // Always startMom (310)
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
      momHighs.push_back(pRight);  // 310
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
    // For baseW=10: [900,920]→slide 10→[910,930], double→[910,950]→slide 20→[930,970], etc.
    std::vector<int> phase2Indices;
    {
      double pLeft = transitionMom;  // 900
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
    // 0. WARMUP: [310, 350] - fixed range, always good, get initial params
    // 1. ANCHOR: First slice of Phase 1 [310, 310+baseW] - use warmup params
    // 2. Phase 0: width doubling below 310 (right edge at 310), uses warmup params
    // 3. Rest of Phase 1 forward from anchor to 900
    // 4. Phase 2 forward with doubling above 900
    // =====================================================
    
    int nSuccess = 0;
    PropagatedParams warmupParams;
    warmupParams.valid = false;
    PropagatedParams anchorParams;
    anchorParams.valid = false;
    
    // Step 0: WARMUP FIT [310, 350] - this always works well!
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
    
    // Step 1: Fit ANCHOR from Phase 1 [310, 310+baseW] using warmup params
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
    
    // Step 2: Fit Phase 0 using WARMUP parameters (width doubling below 310)
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
    // - Phase 0 fits (below 310)
    // - Representative momenta: ~350, ~500, ~700, ~900, ~1200, ~2000
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
    std::vector<double> targetMomenta = {350, 500, 700, 900, 1200, 2000, 3000};
    
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
                           Form("Delta-Beta Fits (Proton) - %s", widthLabels[w].Data()), 1400, 900);
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
  
  TCanvas* cCombDB_1sig = new TCanvas("c_comb_db_1sig", "Combined 1#sigma in #Delta#beta (Proton)", 1000, 800);
  cCombDB_1sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB1 = new TLine(0, 0, 4500, 0);
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
  legDB_1->AddEntry(zeroLineDB1, "#Delta#beta=0 (proton)", "l");
  legDB_1->Draw();
  cCombDB_1sig->Modified();
  cCombDB_1sig->Update();

  // 3σ in Δβ
  TCanvas* cCombDB_3sig = new TCanvas("c_comb_db_3sig", "Combined 3#sigma in #Delta#beta (Proton)", 1000, 800);
  cCombDB_3sig->cd();
  h2DB->Draw("colz");
  
  TLine* zeroLineDB3 = new TLine(0, 0, 4500, 0);
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
  legDB_3->AddEntry(zeroLineDB3, "#Delta#beta=0 (proton)", "l");
  legDB_3->Draw();
  cCombDB_3sig->Modified();
  cCombDB_3sig->Update();

  // =====================================================
  // TRANSFORM TO (p, β) SPACE
  // =====================================================
  
  // 1σ in (p, β)
  TCanvas* cCombPB_1sig = new TCanvas("c_comb_pb_1sig", "Combined 1#sigma in (p, #beta) Proton", 1000, 800);
  cCombPB_1sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* protonCurvePB = new TF1("protonCurvePB", "x/sqrt(x*x + 938.272*938.272)", 0, 4500);
  protonCurvePB->SetLineColor(kBlack);
  protonCurvePB->SetLineStyle(kDashed);
  protonCurvePB->SetLineWidth(2);
  protonCurvePB->Draw("same");
  
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
      double beta_proton = betaProton(p);
      double beta_mean = beta_proton + smoothMean[i];
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 1.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPointsUpper.push_back(p);
        betaPointsUpper.push_back(beta_upper);
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 1.0 * smoothSigma[i];
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
  legPB_1->AddEntry(protonCurvePB, "m_{p}=938.27", "l");
  legPB_1->Draw();
  cCombPB_1sig->Modified();
  cCombPB_1sig->Update();

  // 3σ and 5σ in (p, β)
  TCanvas* cCombPB_3sig = new TCanvas("c_comb_pb_3sig", "Combined 3#sigma and 5#sigma in (p, #beta) Proton", 1000, 800);
  cCombPB_3sig->cd();
  if (h2PB) h2PB->Draw("colz");
  
  TF1* protonCurvePB3 = new TF1("protonCurvePB3", "x/sqrt(x*x + 938.272*938.272)", 0, 4500);
  protonCurvePB3->SetLineColor(kBlack);
  protonCurvePB3->SetLineStyle(kDashed);
  protonCurvePB3->SetLineWidth(2);
  protonCurvePB3->Draw("same");
  
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
      double beta_proton = betaProton(p);
      double beta_mean = beta_proton + smoothMean[i];
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 3.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints3Upper.push_back(p);
        betaPoints3Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 3.0 * smoothSigma[i];
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 5.0 * smoothSigma[i];
      if (beta_upper > 0.1 && beta_upper < 1.15) {
        pPoints5Upper.push_back(p);
        betaPoints5Upper.push_back(beta_upper);
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 5.0 * smoothSigma[i];
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
  legPB_3->AddEntry(protonCurvePB3, "m_{p}=938.27", "l");
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
  
  const double protonMass2 = gProtonMass * gProtonMass;  // ~880,354 MeV²/c⁴
  
  // 1σ in (p, mass²)
  TCanvas* cCombM2_1sig = new TCanvas("c_comb_m2_1sig", "Combined 1#sigma in (p, mass^{2}) Proton", 1000, 800);
  cCombM2_1sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  // Draw proton mass² line
  TLine* protonLineM2_1 = new TLine(0, protonMass2, 4500, protonMass2);
  protonLineM2_1->SetLineColor(kBlack);
  protonLineM2_1->SetLineStyle(kDashed);
  protonLineM2_1->SetLineWidth(2);
  protonLineM2_1->Draw("same");
  
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 1.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > 0 && m2_upper < 2000000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 1.0 * smoothSigma[i];
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
  legM2_1->AddEntry(protonLineM2_1, Form("m_{p}^{2}=%.0f", protonMass2), "l");
  legM2_1->Draw();
  cCombM2_1sig->Modified();
  cCombM2_1sig->Update();

  // 3σ in (p, mass²)
  TCanvas* cCombM2_3sig = new TCanvas("c_comb_m2_3sig", "Combined 3#sigma in (p, mass^{2}) Proton", 1000, 800);
  cCombM2_3sig->cd();
  if (h2M2) h2M2->Draw("colz");
  
  // Draw proton mass² line
  TLine* protonLineM2_3 = new TLine(0, protonMass2, 4500, protonMass2);
  protonLineM2_3->SetLineColor(kBlack);
  protonLineM2_3->SetLineStyle(kDashed);
  protonLineM2_3->SetLineWidth(2);
  protonLineM2_3->Draw("same");
  
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 3.0 * smoothSigma[i];
      double m2_upper = mass2FromPBeta(p, beta_upper);
      if (m2_upper > 0 && m2_upper < 2000000) {
        pPointsUpper.push_back(p);
        m2PointsUpper.push_back(m2_upper);
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 3.0 * smoothSigma[i];
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
  legM2_3->AddEntry(protonLineM2_3, Form("m_{p}^{2}=%.0f", protonMass2), "l");
  legM2_3->Draw();
  cCombM2_3sig->Modified();
  cCombM2_3sig->Update();

  // =====================================================
  // TRANSFORM TO (mass, a) SPACE
  // =====================================================
  
  TCanvas* cCombMA_1sig = new TCanvas("c_comb_ma_1sig", "Combined 1#sigma in (mass, a) Proton", 1000, 800);
  cCombMA_1sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* protonLineMA = new TLine(gProtonMass, 0, gProtonMass, 16);
  protonLineMA->SetLineColor(kBlack);
  protonLineMA->SetLineStyle(kDashed);
  protonLineMA->SetLineWidth(2);
  protonLineMA->Draw("same");
  
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 1.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > 0 && mass_u < 2000 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 1.0 * smoothSigma[i];
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
  legMA_1->AddEntry(protonLineMA, "m_{p}=938.27", "l");
  legMA_1->Draw();
  cCombMA_1sig->Modified();
  cCombMA_1sig->Update();

  // 3σ in (mass, a)
  TCanvas* cCombMA_3sig = new TCanvas("c_comb_ma_3sig", "Combined 3#sigma in (mass, a) Proton", 1000, 800);
  cCombMA_3sig->cd();
  if (h2MA) h2MA->Draw("colz");
  
  TLine* protonLineMA3 = new TLine(gProtonMass, 0, gProtonMass, 16);
  protonLineMA3->SetLineColor(kBlack);
  protonLineMA3->SetLineStyle(kDashed);
  protonLineMA3->SetLineWidth(2);
  protonLineMA3->Draw("same");
  
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
      double beta_proton = betaProton(p);
      
      double beta_upper = beta_proton + smoothMean[i] + 3.0 * smoothSigma[i];
      double mass_u, a_u;
      if (beta_upper > 0.1 && beta_upper < 1.5) {
        if (pBetaToMassA(p, beta_upper, gSSquared, mass_u, a_u) && 
            mass_u > 0 && mass_u < 2000 && a_u > 0 && a_u < 16) {
          massPointsUpper.push_back(mass_u);
          aPointsUpper.push_back(a_u);
        }
      }
      
      double beta_lower = beta_proton + smoothMean[i] - 3.0 * smoothSigma[i];
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
  legMA_3->AddEntry(protonLineMA3, "m_{p}=938.27", "l");
  legMA_3->Draw();
  cCombMA_3sig->Modified();
  cCombMA_3sig->Update();

  // =====================================================
  // PARAMETER PLOTS
  // =====================================================
  
  TCanvas* cPar = new TCanvas("c_par", "Fit Parameters vs Momentum (Proton)", 1200, 800);
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
  TLine* zeroMean = new TLine(0, 0, 4500, 0);
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
  TLine* chi2Line = new TLine(0, 1, 4500, 1);
  chi2Line->SetLineColor(kRed);
  chi2Line->SetLineStyle(kDashed);
  chi2Line->Draw("same");
  legChi2->Draw();

  // =====================================================
  // ADDITION #6: GENERATE AND SAVE TCutG OBJECTS
  // =====================================================
  
  cout << "\n=============================================" << endl;
  cout << "=== GENERATING TCutG OBJECTS ===" << endl;
  cout << "=============================================" << endl;
  
  // SELECT WHICH WIDTH TO USE FOR FINAL CUTS
  // Options: 0 = 1x(10), 1 = 2x(20), 2 = 4x(40), 3 = 8x(80)
  // Recommendation: 1 (2x) balances resolution and statistics
  int selectedWidthForCuts = 1;  // <-- CHANGE THIS TO SELECT DIFFERENT WIDTH
  
  cout << "Using width: " << widthLabels[selectedWidthForCuts] << endl;
  
  // Generate smoothed contours and TCutG
  ContourOutput contours = generateSmoothedCuts(
    allFitResults[selectedWidthForCuts],  // Fit results for selected width
    allMomCenters[selectedWidthForCuts],  // Momentum centers
    selectedWidthForCuts,                  // Width index
    gProtonMass,                           // Particle mass (938.272 for proton)
    0.0,                                   // pMin [MeV/c] - will be clipped to data
    4500.0,                                // pMax [MeV/c] - will be clipped to data
    5.0,                                   // pStep for TCutG sampling [MeV/c]
    "p",                                   // Particle name prefix
    "p_p",                                 // TTree branch name for momentum
    "p_beta",                              // TTree branch name for beta
    5,                                     // Median filter window
    7                                      // Gaussian smoothing window
  );
  
  // Save everything to ROOT file
  if (contours.valid) {
    saveContourResults(
      contours,
      allFitResults,
      allMomCenters,
      widthLabels,
      nWidths,
      "p_pid_cuts.root",                   // Output filename
      "p"                                  // Particle name
    );
    
    // Draw cuts on histogram
    TCanvas* cCuts = drawCutsOnHistogram(h2PB, contours, "c_p_cuts", gProtonMass);
    if (cCuts) {
      cCuts->SaveAs("p_pid_cuts_overlay.png");
    }
  }
  // =====================================================
  // END OF TCutG GENERATION
  // =====================================================

  // =====================================================
  // SAVE OUTPUT
  // =====================================================
  
  for (int w = 0; w < nWidths; ++w) {
    cFits[w]->SaveAs(Form("proton_fits_dBeta_%s.png", widthLabels[w].Data()));
  }
  cCombDB_1sig->SaveAs("proton_dBeta_combined_1sigma.png");
  cCombDB_3sig->SaveAs("proton_dBeta_combined_3sigma.png");
  cCombPB_1sig->SaveAs("proton_pbeta_combined_1sigma.png");
  cCombPB_3sig->SaveAs("proton_pbeta_combined_3sigma.png");
  cCombM2_1sig->SaveAs("proton_mass2_combined_1sigma.png");
  cCombM2_3sig->SaveAs("proton_mass2_combined_3sigma.png");
  cCombMA_1sig->SaveAs("proton_massa_combined_1sigma.png");
  cCombMA_3sig->SaveAs("proton_massa_combined_3sigma.png");
  cPar->SaveAs("proton_parameters_dBeta.png");

  // Output file
  std::ofstream outfile("proton_fit_results_dBeta.txt");
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

  cout << "\n=== PROTON Analysis Complete ===" << endl;
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
  cout << "\n=== TCutG OUTPUT ===" << endl;
  cout << "  ROOT file: p_pid_cuts.root" << endl;
  cout << "  Contains: FitResults TTree, Contours/, TCutG/" << endl;
  cout << "\nOutput files saved." << endl;
}
