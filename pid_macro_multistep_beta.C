// pid_macro_sliding_window.C
// Performs 4 parallel sliding window scans with different widths
// Regular fits up to a = 5, then ONE final wide slice from a=5 to a=15
// This captures all remaining high-momentum pions in sparse region

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

// Global parameters
const double gSSquared = 1e-4;
const double gRegularMaxA = 5.0;    // Regular fitting stops here
const double gFinalSliceEnd = 15.0; // Final wide slice extends to here

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
  double windowWidth;
  double lowerBound;
  double upperBound;
  bool isFinalSlice;  // Flag for the final wide slice
};

// Convert (mass, a) -> (p, beta)
bool massAToMomBeta(double mass, double a, double s2, double& p_out, double& beta_out) {
  double m2 = mass * mass;
  double m4 = m2 * m2;
  double a2 = a * a;
  
  double A = s2;
  double B = 1.0 - a2 - s2 * m2;
  double D = -m4;
  
  if (std::abs(A) < 1e-15) {
    if (std::abs(B) < 1e-15) return false;
    double x = -D / B;
    if (x < m2) return false;
    p_out = std::sqrt(x - m2);
    beta_out = p_out / std::sqrt(x);
    return true;
  }
  
  double b = B / A;
  double c = 0.0;
  double d = D / A;
  
  double p = c - b*b/3.0;
  double q = 2.0*b*b*b/27.0 - b*c/3.0 + d;
  
  double discriminant = q*q/4.0 + p*p*p/27.0;
  
  double x = 0;
  
  if (discriminant >= 0) {
    double sqrtD = std::sqrt(discriminant);
    double u = std::cbrt(-q/2.0 + sqrtD);
    double v = std::cbrt(-q/2.0 - sqrtD);
    x = u + v - b/3.0;
  } else {
    double r = std::sqrt(-p*p*p/27.0);
    double phi = std::acos(-q/(2.0*r));
    double rCbrt = std::cbrt(r);
    
    double x1 = 2.0*rCbrt*std::cos(phi/3.0) - b/3.0;
    double x2 = 2.0*rCbrt*std::cos((phi + 2.0*M_PI)/3.0) - b/3.0;
    double x3 = 2.0*rCbrt*std::cos((phi + 4.0*M_PI)/3.0) - b/3.0;
    
    x = -1e9;
    if (x1 > m2 && x1 > x) x = x1;
    if (x2 > m2 && x2 > x) x = x2;
    if (x3 > m2 && x3 > x) x = x3;
  }
  
  if (x < m2 || x < 0) return false;
  
  p_out = std::sqrt(x - m2);
  beta_out = p_out / std::sqrt(x);
  
  if (beta_out <= 0 || beta_out >= 1.5 || p_out < 0) return false;
  
  return true;
}

// Fit with polynomial background selection
FitResult tryFitWithPolynomials(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx, 
                                 double windowWidth, double lowerBound, double upperBound, bool isFinal) {
  FitResult best;
  best.success = false;
  best.mean = 139.57;
  best.sigma = 10.0;
  best.amplitude = 0.0;
  best.polyOrder = 1;
  best.chi2ndf = 1e9;
  best.entries = proj ? proj->GetEntries() : 0;
  best.windowWidth = windowWidth;
  best.lowerBound = lowerBound;
  best.upperBound = upperBound;
  best.isFinalSlice = isFinal;

  if (!proj || best.entries < 20) return best;

  int bin_min = proj->FindFixBin(fitMin);
  int bin_max = proj->FindFixBin(fitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 5.0) return best;

  int maxBin = 0;
  double maxVal = -1.0;
  for (int b = bin_min; b <= bin_max; ++b) {
    double v = proj->GetBinContent(b);
    if (v > maxVal) { maxVal = v; maxBin = b; }
  }
  double peakPos = (maxBin > 0) ? proj->GetBinCenter(maxBin) : 139.57;

  for (int polyOrder = 0; polyOrder <= 4; ++polyOrder) {
    TString funcName = Form("tryfit_w%d_s%d_p%d", widthIdx, sliceIdx, polyOrder);
    TString funcExpr;
    int nBkgParams = polyOrder + 1;
    
    switch (polyOrder) {
      case 0: funcExpr = "gaus(0) + [3]"; break;
      case 1: funcExpr = "gaus(0) + pol1(3)"; break;
      case 2: funcExpr = "gaus(0) + pol2(3)"; break;
      case 3: funcExpr = "gaus(0) + pol3(3)"; break;
      case 4: funcExpr = "gaus(0) + pol4(3)"; break;
    }

    TF1* fitFunc = new TF1(funcName, funcExpr, fitMin, fitMax);
    
    fitFunc->SetParameter(0, std::max(1.0, maxVal * 0.8));
    fitFunc->SetParameter(1, peakPos);
    fitFunc->SetParameter(2, 10.0);
    
    double bkgLevel = proj->GetBinContent(bin_min);
    fitFunc->SetParameter(3, bkgLevel);
    for (int pp = 1; pp < nBkgParams; ++pp)
      fitFunc->SetParameter(3 + pp, 0.0);

    fitFunc->SetParLimits(0, 0.0, std::max(10.0, 2.0 * maxVal + 1.0));
    fitFunc->SetParLimits(1, 100.0, 180.0);
    fitFunc->SetParLimits(2, 2.0, 40.0);

    TFitResultPtr result = proj->Fit(fitFunc, "SQR0");

    bool valid = false;
    double chi2ndf = 1e9;

    if (result.Get() && result->IsValid() && result->Status() == 0) {
      double fMean = fitFunc->GetParameter(1);
      double fSigma = fitFunc->GetParameter(2);
      double fAmp = fitFunc->GetParameter(0);

      if (fMean > 115.0 && fMean < 165.0 && 
          fSigma > 2.0 && fSigma < 40.0 && 
          fAmp > 0.5) {
        valid = true;
        chi2ndf = (result->Ndf() > 0) ? result->Chi2() / result->Ndf() : 1e6;
      }
    }

    if (valid && chi2ndf < best.chi2ndf) {
      best.success = true;
      best.mean = fitFunc->GetParameter(1);
      best.sigma = fitFunc->GetParameter(2);
      best.amplitude = fitFunc->GetParameter(0);
      best.polyOrder = polyOrder;
      best.chi2ndf = chi2ndf;
      best.bkgParams.clear();
      for (int pp = 0; pp < nBkgParams; ++pp)
        best.bkgParams.push_back(fitFunc->GetParameter(3 + pp));
    }

    delete fitFunc;
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

  const double sSquared = gSSquared;

  // --- 2D histograms - Y range to 16 to see everything
  const char* h2name = "h2_pid_mass_vs_a";
  TString drawCmd = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) * sign(pim_p) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) * sign(pim_p) >> %s(300,0,300,800,0,16)",
    sSquared, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "isBest==1", "colz");
  TH2F* h2 = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2) { cout << "Failed to create histogram" << endl; return; }

  h2->SetTitle(Form("Mass vs a-parameter (s^{2} = %.1e)", sSquared));
  h2->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
  h2->GetYaxis()->SetTitle("a * sign(p)");

  // p vs beta
  const char* h2PBname = "h2_mom_vs_beta";
  if (gDirectory->FindObject(h2PBname)) gDirectory->Delete(Form("%s;*", h2PBname));
  chain->Draw(Form("pim_beta : pim_p >> %s(400,0,2000,400,0.4,1.1)", h2PBname), "isBest==1", "colz");
  TH2F* h2PB = static_cast<TH2F*>(gDirectory->Get(h2PBname));
  if (h2PB) {
    h2PB->SetTitle("Momentum vs #beta");
    h2PB->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2PB->GetYaxis()->SetTitle("#beta");
  }

  // p vs m²
  const char* h2PM2name = "h2_mom_vs_mass2";
  if (gDirectory->FindObject(h2PM2name)) gDirectory->Delete(Form("%s;*", h2PM2name));
  chain->Draw(Form("pim_p*pim_p*(1.0/(pim_beta*pim_beta) - 1.0) : pim_p >> %s(400,0,2000,400,-10000,80000)", h2PM2name), 
              "isBest==1", "colz");
  TH2F* h2PM2 = static_cast<TH2F*>(gDirectory->Get(h2PM2name));
  if (h2PM2) {
    h2PM2->SetTitle("Momentum vs Mass^{2}");
    h2PM2->GetXaxis()->SetTitle("Momentum [MeV/c]");
    h2PM2->GetYaxis()->SetTitle("Mass^{2} [MeV^{2}/c^{4}]");
  }

  // --- Parameters
  const double baseWidth = 0.2;
  const double stepSize = 0.01;
  const double startLower = 0.3;
  const double regularMaxA = gRegularMaxA;  // Regular fits up to here
  const double finalSliceEnd = gFinalSliceEnd;  // Final slice extends to here
  
  const double fitRangeMin = 80.0;
  const double fitRangeMax = 220.0;

  // Width multipliers for regular fits
  const int nWidths = 4;
  double widthMultipliers[nWidths] = {1.0, 2.0, 4.0, 8.0};
  TString widthLabels[nWidths] = {"1x", "2x", "4x", "8x"};
  int widthColors[nWidths] = {kBlue, kGreen+2, kOrange+1, kRed};

  // For p-beta plots (4x and 8x)
  int pbIndices[2] = {2, 3};

  // --- Storage
  std::vector<std::vector<double>> allSliceCenters(nWidths);
  std::vector<std::vector<FitResult>> allFitResults(nWidths);

  // --- Create fit canvases (12 regular + 1 final = show 12 with last being final)
  TCanvas* cFits[nWidths];
  const int nDisplayPerWidth = 12;
  
  for (int w = 0; w < nWidths; ++w) {
    cFits[w] = new TCanvas(Form("c_fit_%d", w),
                           Form("Pion Fits - Width %s (regular + final slice)", widthLabels[w].Data()),
                           1200, 800);
    cFits[w]->Divide(4, 3);
  }

  // --- Perform scans
  for (int w = 0; w < nWidths; ++w) {
    double currentWidth = baseWidth * widthMultipliers[w];
    
    cout << "\n=============================================" << endl;
    cout << "=== Width " << widthLabels[w] << " (" << currentWidth << ") ===" << endl;
    cout << "=== Regular fits: a = " << startLower << " to " << regularMaxA << endl;
    cout << "=== Final slice: a = " << regularMaxA << " to " << finalSliceEnd << endl;
    cout << "=============================================" << endl;

    // Build slice list for regular fits
    std::vector<double> centers;
    std::vector<double> lowerBounds;
    std::vector<double> upperBounds;
    std::vector<double> widths;
    std::vector<bool> isFinal;
    
    // Regular slices
    double currentA = startLower;
    while (currentA + currentWidth <= regularMaxA + 0.001) {
      double lower = currentA;
      double upper = currentA + currentWidth;
      double center = 0.5 * (lower + upper);
      
      lowerBounds.push_back(lower);
      upperBounds.push_back(upper);
      centers.push_back(center);
      widths.push_back(currentWidth);
      isFinal.push_back(false);
      
      currentA += stepSize;
    }
    
    // Add ONE final wide slice from regularMaxA to finalSliceEnd
    double finalWidth = finalSliceEnd - regularMaxA;
    double finalCenter = 0.5 * (regularMaxA + finalSliceEnd);
    lowerBounds.push_back(regularMaxA);
    upperBounds.push_back(finalSliceEnd);
    centers.push_back(finalCenter);
    widths.push_back(finalWidth);
    isFinal.push_back(true);
    
    int nSlices = centers.size();
    int nRegular = nSlices - 1;
    
    cout << "Regular slices: " << nRegular << ", plus 1 final slice" << endl;
    cout << "Total slices: " << nSlices << endl;
    cout << "Final slice: a = [" << regularMaxA << ", " << finalSliceEnd << "], width = " << finalWidth << endl;

    allSliceCenters[w] = centers;
    allFitResults[w].resize(nSlices);

    // Display indices - 11 regular evenly spaced + 1 final
    std::vector<int> displayIndices;
    for (int d = 0; d < nDisplayPerWidth - 1; ++d) {
      int idx = (nRegular > 1) ? (int)((double)d * (nRegular - 1) / (nDisplayPerWidth - 2) + 0.5) : 0;
      if (idx >= nRegular) idx = nRegular - 1;
      displayIndices.push_back(idx);
    }
    displayIndices.push_back(nSlices - 1);  // Always show final slice
    
    cout << "Display indices: ";
    for (int idx : displayIndices) {
      cout << idx << "(a=" << centers[idx] << (isFinal[idx] ? ",FINAL" : "") << ") ";
    }
    cout << endl;

    // Perform all fits
    int nSuccess = 0, nSuccessRegular = 0;
    for (int i = 0; i < nSlices; ++i) {
      int ybin_lo = h2->GetYaxis()->FindFixBin(lowerBounds[i] + 1e-6);
      int ybin_hi = h2->GetYaxis()->FindFixBin(upperBounds[i] - 1e-6);
      
      TString projName = Form("proj_fit_w%d_%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2->ProjectionX(projName, ybin_lo, ybin_hi, "e");

      allFitResults[w][i] = tryFitWithPolynomials(proj, fitRangeMin, fitRangeMax, i, w, 
                                                   widths[i], lowerBounds[i], upperBounds[i], isFinal[i]);
      
      if (allFitResults[w][i].success) {
        nSuccess++;
        if (!isFinal[i]) nSuccessRegular++;
      }

      if (i < 5 || i % 50 == 0 || i == nSlices - 1 || isFinal[i]) {
        cout << Form("  Slice %3d: a=[%.2f,%.2f], width=%.2f, entries=%.0f, fit=%s%s", 
                     i, lowerBounds[i], upperBounds[i], widths[i], 
                     allFitResults[w][i].entries,
                     allFitResults[w][i].success ? Form("OK(pol%d)", allFitResults[w][i].polyOrder) : "FAIL",
                     isFinal[i] ? " [FINAL SLICE]" : "") << endl;
      }
      delete proj;
    }
    
    cout << "Success: " << nSuccess << "/" << nSlices 
         << " (regular: " << nSuccessRegular << "/" << nRegular << ", final: " 
         << (allFitResults[w][nSlices-1].success ? "OK" : "FAIL") << ")" << endl;

    // Display fits
    for (size_t d = 0; d < displayIndices.size(); ++d) {
      int i = displayIndices[d];
      FitResult& result = allFitResults[w][i];

      cFits[w]->cd(d + 1);
      gPad->SetLeftMargin(0.12);
      gPad->SetRightMargin(0.04);
      gPad->SetTopMargin(0.10);
      gPad->SetBottomMargin(0.12);

      int ybin_lo = h2->GetYaxis()->FindFixBin(lowerBounds[i] + 1e-6);
      int ybin_hi = h2->GetYaxis()->FindFixBin(upperBounds[i] - 1e-6);
      
      TString projName = Form("proj_disp_w%d_d%d", w, (int)d);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2->ProjectionX(projName, ybin_lo, ybin_hi, "e");
      if (!proj) continue;

      // Title - mark final slice specially
      TString title;
      if (isFinal[i]) {
        title = Form("FINAL: %.1f < a < %.1f", lowerBounds[i], upperBounds[i]);
      } else {
        title = Form("%.2f < a < %.2f", lowerBounds[i], upperBounds[i]);
      }
      proj->SetTitle(title);
      proj->GetXaxis()->SetRangeUser(fitRangeMin, fitRangeMax);
      proj->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
      proj->GetYaxis()->SetTitle("Counts");
      proj->SetLineColor(kBlack);
      proj->SetMarkerStyle(20);
      proj->SetMarkerSize(0.4);
      proj->Draw("E");

      if (result.success) {
        TString funcExpr;
        switch (result.polyOrder) {
          case 0: funcExpr = "gaus(0) + [3]"; break;
          case 1: funcExpr = "gaus(0) + pol1(3)"; break;
          case 2: funcExpr = "gaus(0) + pol2(3)"; break;
          case 3: funcExpr = "gaus(0) + pol3(3)"; break;
          case 4: funcExpr = "gaus(0) + pol4(3)"; break;
        }
        
        TF1* fitFunc = new TF1(Form("fitdisp_w%d_d%d", w, (int)d), funcExpr, fitRangeMin, fitRangeMax);
        fitFunc->SetParameter(0, result.amplitude);
        fitFunc->SetParameter(1, result.mean);
        fitFunc->SetParameter(2, result.sigma);
        for (size_t p = 0; p < result.bkgParams.size(); ++p)
          fitFunc->SetParameter(3 + p, result.bkgParams[p]);
        fitFunc->SetLineColor(kRed);
        fitFunc->SetLineWidth(2);
        fitFunc->Draw("same");

        TF1* sig = new TF1(Form("sig_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
        sig->SetParameters(result.amplitude, result.mean, result.sigma);
        sig->SetLineColor(kBlue);
        sig->SetLineStyle(kDashed);
        sig->SetLineWidth(2);
        sig->Draw("same");

        TF1* bkg = new TF1(Form("bkg_w%d_d%d", w, (int)d), getBkgFuncExpr(result.polyOrder), fitRangeMin, fitRangeMax);
        for (size_t p = 0; p < result.bkgParams.size(); ++p)
          bkg->SetParameter(p, result.bkgParams[p]);
        bkg->SetLineColor(kGreen+2);
        bkg->SetLineStyle(kDashed);
        bkg->SetLineWidth(2);
        bkg->Draw("same");
      }

      TLatex tex;
      tex.SetNDC();
      tex.SetTextSize(0.040);
      if (isFinal[i]) {
        tex.SetTextColor(kMagenta+1);
        tex.DrawLatex(0.50, 0.82, "FINAL SLICE");
        tex.SetTextColor(kBlack);
      }
      if (result.success) {
        tex.DrawLatex(0.50, 0.74, Form("Mean=%.1f", result.mean));
        tex.DrawLatex(0.50, 0.66, Form("#sigma=%.1f", result.sigma));
        tex.DrawLatex(0.50, 0.58, Form("pol%d, #chi^{2}=%.1f", result.polyOrder, result.chi2ndf));
      } else {
        tex.SetTextColor(kRed);
        tex.DrawLatex(0.50, 0.74, "Fit failed");
        tex.SetTextColor(kBlack);
      }
      tex.DrawLatex(0.50, 0.50, Form("N=%.0f", result.entries));
      
      gPad->Modified();
      gPad->Update();
    }
    cFits[w]->Modified();
    cFits[w]->Update();
  }

  // --- Individual contour plots
  TCanvas* cContours[nWidths];
  for (int w = 0; w < nWidths; ++w) {
    cContours[w] = new TCanvas(Form("c_contour_%d", w),
                               Form("Pion Selection - %s", widthLabels[w].Data()), 800, 600);
    cContours[w]->cd();
    h2->Draw("colz");

    std::vector<double> massPoints[3], aPoints[3];
    int nSlices = allSliceCenters[w].size();
    
    // Forward pass
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          massPoints[s].push_back(allFitResults[w][i].mean + (s+1) * allFitResults[w][i].sigma);
          aPoints[s].push_back(allSliceCenters[w][i]);
        }
      }
    }
    // Backward pass
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          massPoints[s].push_back(allFitResults[w][i].mean - (s+1) * allFitResults[w][i].sigma);
          aPoints[s].push_back(allSliceCenters[w][i]);
        }
      }
    }

    int contColors[3] = {kBlue, kGreen+2, kRed};
    TGraph* graphs[3] = {nullptr};
    for (int s = 0; s < 3; ++s) {
      if (!massPoints[s].empty()) {
        graphs[s] = new TGraph(massPoints[s].size(), massPoints[s].data(), aPoints[s].data());
        graphs[s]->SetLineColor(contColors[s]);
        graphs[s]->SetLineWidth(6);
        graphs[s]->Draw("L");
      }
    }

    // Draw transition line at regularMaxA
    TLine* transLine = new TLine(0, regularMaxA, 300, regularMaxA);
    transLine->SetLineColor(kMagenta);
    transLine->SetLineStyle(kDashed);
    transLine->SetLineWidth(2);
    transLine->Draw("same");

    TLegend* leg = new TLegend(0.65, 0.60, 0.88, 0.88);
    leg->SetHeader(Form("Width: %s", widthLabels[w].Data()));
    if (graphs[0]) leg->AddEntry(graphs[0], "1#sigma", "l");
    if (graphs[1]) leg->AddEntry(graphs[1], "2#sigma", "l");
    if (graphs[2]) leg->AddEntry(graphs[2], "3#sigma", "l");
    leg->AddEntry(transLine, Form("a=%.0f (final slice)", regularMaxA), "l");
    leg->Draw();
    cContours[w]->Modified();
    cContours[w]->Update();
  }

  // --- Combined 3-sigma plot
  TCanvas* cCombined3 = new TCanvas("c_combined_3sigma", "Contour Comparison - All Widths (3#sigma)", 1000, 800);
  cCombined3->cd();
  h2->Draw("colz");
  
  TLine* transLine3 = new TLine(0, regularMaxA, 300, regularMaxA);
  transLine3->SetLineColor(kMagenta);
  transLine3->SetLineStyle(kDashed);
  transLine3->SetLineWidth(2);
  transLine3->Draw("same");
  
  TLegend* legComb3 = new TLegend(0.65, 0.55, 0.88, 0.88);
  legComb3->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> massPoints, aPoints;
    int nSlices = allSliceCenters[w].size();
    for (int i = 0; i < nSlices; ++i)
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean + 3.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    for (int i = nSlices - 1; i >= 0; --i)
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean - 3.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    if (!massPoints.empty()) {
      TGraph* g = new TGraph(massPoints.size(), massPoints.data(), aPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(6);
      g->Draw("L");
      legComb3->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legComb3->AddEntry(transLine3, Form("a=%.0f (final slice)", regularMaxA), "l");
  legComb3->Draw();
  cCombined3->Modified();
  cCombined3->Update();

  // --- Combined 1-sigma plot
  TCanvas* cCombined1 = new TCanvas("c_combined_1sigma", "Contour Comparison - All Widths (1#sigma)", 1000, 800);
  cCombined1->cd();
  h2->Draw("colz");
  
  TLine* transLine1 = new TLine(0, regularMaxA, 300, regularMaxA);
  transLine1->SetLineColor(kMagenta);
  transLine1->SetLineStyle(kDashed);
  transLine1->SetLineWidth(2);
  transLine1->Draw("same");
  
  TLegend* legComb1 = new TLegend(0.65, 0.55, 0.88, 0.88);
  legComb1->SetHeader("1#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> massPoints, aPoints;
    int nSlices = allSliceCenters[w].size();
    for (int i = 0; i < nSlices; ++i)
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean + 1.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    for (int i = nSlices - 1; i >= 0; --i)
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean - 1.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    if (!massPoints.empty()) {
      TGraph* g = new TGraph(massPoints.size(), massPoints.data(), aPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(6);
      g->Draw("L");
      legComb1->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  legComb1->AddEntry(transLine1, Form("a=%.0f (final slice)", regularMaxA), "l");
  legComb1->Draw();
  cCombined1->Modified();
  cCombined1->Update();

  // --- (p, beta) plot with cuts (4x and 8x)
  TCanvas* cPBeta = new TCanvas("c_p_beta", "Momentum vs #beta with PID cuts", 1000, 800);
  cPBeta->cd();
  if (h2PB) h2PB->Draw("colz");

  TLegend* legPB = new TLegend(0.12, 0.70, 0.40, 0.88);
  legPB->SetHeader("3#sigma cuts");

  for (int idx = 0; idx < 2; ++idx) {
    int w = pbIndices[idx];
    std::vector<double> pPoints, betaPoints;
    int nSlices = allSliceCenters[w].size();
    
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        double mass = allFitResults[w][i].mean + 3.0 * allFitResults[w][i].sigma;
        double a = allSliceCenters[w][i];
        double p, beta;
        if (massAToMomBeta(mass, a, sSquared, p, beta)) {
          pPoints.push_back(p);
          betaPoints.push_back(beta);
        }
      }
    }
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        double mass = allFitResults[w][i].mean - 3.0 * allFitResults[w][i].sigma;
        double a = allSliceCenters[w][i];
        double p, beta;
        if (massAToMomBeta(mass, a, sSquared, p, beta)) {
          pPoints.push_back(p);
          betaPoints.push_back(beta);
        }
      }
    }

    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), betaPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(6);
      g->Draw("L");
      legPB->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  
  TF1* pionLinePB = new TF1("pionLinePB", "x/sqrt(x*x + 139.57*139.57)", 0, 2000);
  pionLinePB->SetLineColor(kBlack);
  pionLinePB->SetLineStyle(kDashed);
  pionLinePB->SetLineWidth(2);
  pionLinePB->Draw("same");
  legPB->AddEntry(pionLinePB, "m_{#pi}=139.57", "l");
  legPB->Draw();
  cPBeta->Modified();
  cPBeta->Update();

  // --- (p, m²) plot with cuts
  TCanvas* cPM2 = new TCanvas("c_p_mass2", "Momentum vs Mass^{2} with PID cuts", 1000, 800);
  cPM2->cd();
  if (h2PM2) h2PM2->Draw("colz");

  TLegend* legPM2 = new TLegend(0.12, 0.70, 0.40, 0.88);
  legPM2->SetHeader("3#sigma cuts");

  for (int idx = 0; idx < 2; ++idx) {
    int w = pbIndices[idx];
    std::vector<double> pPoints, m2Points;
    int nSlices = allSliceCenters[w].size();
    
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        double mass = allFitResults[w][i].mean + 3.0 * allFitResults[w][i].sigma;
        double a = allSliceCenters[w][i];
        double p, beta;
        if (massAToMomBeta(mass, a, sSquared, p, beta)) {
          double m2 = p * p * (1.0 / (beta * beta) - 1.0);
          pPoints.push_back(p);
          m2Points.push_back(m2);
        }
      }
    }
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        double mass = allFitResults[w][i].mean - 3.0 * allFitResults[w][i].sigma;
        double a = allSliceCenters[w][i];
        double p, beta;
        if (massAToMomBeta(mass, a, sSquared, p, beta)) {
          double m2 = p * p * (1.0 / (beta * beta) - 1.0);
          pPoints.push_back(p);
          m2Points.push_back(m2);
        }
      }
    }

    if (!pPoints.empty()) {
      TGraph* g = new TGraph(pPoints.size(), pPoints.data(), m2Points.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(6);
      g->Draw("L");
      legPM2->AddEntry(g, Form("%s", widthLabels[w].Data()), "l");
    }
  }
  
  TLine* pionLineM2 = new TLine(0, 139.57*139.57, 2000, 139.57*139.57);
  pionLineM2->SetLineColor(kBlack);
  pionLineM2->SetLineStyle(kDashed);
  pionLineM2->SetLineWidth(2);
  pionLineM2->Draw("same");
  legPM2->AddEntry(pionLineM2, "m_{#pi}^{2}=19480", "l");
  legPM2->Draw();
  cPM2->Modified();
  cPM2->Update();

  // --- Parameter plots
  TCanvas* cPar = new TCanvas("c_par", "Pion Parameters vs. a", 1000, 450);
  cPar->Divide(2, 1);

  cPar->cd(1);
  gPad->SetLeftMargin(0.12);
  TMultiGraph* mgMean = new TMultiGraph();
  TLegend* legMean = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i)
      if (allFitResults[w][i].success)
        g->SetPoint(np++, allSliceCenters[w][i], allFitResults[w][i].mean);
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.5);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgMean->Add(g, "LP");
      legMean->AddEntry(g, Form("%s", widthLabels[w].Data()), "lp");
    }
  }
  mgMean->SetTitle("Pion Mass vs. a;a-parameter;Mean [MeV/c^{2}]");
  mgMean->Draw("A");
  mgMean->GetXaxis()->SetLimits(0.0, 12.0);
  mgMean->GetYaxis()->SetRangeUser(120.0, 160.0);
  
  TF1* constLine = new TF1("cl", "139.57", 0, 12);
  constLine->SetLineColor(kBlack);
  constLine->SetLineStyle(kDashed);
  constLine->Draw("same");
  
  TLine* transLineMean = new TLine(regularMaxA, 120, regularMaxA, 160);
  transLineMean->SetLineColor(kMagenta);
  transLineMean->SetLineStyle(kDashed);
  transLineMean->Draw("same");
  
  legMean->AddEntry(constLine, "m_{#pi}=139.57", "l");
  legMean->Draw();

  cPar->cd(2);
  gPad->SetLeftMargin(0.12);
  TMultiGraph* mgSigma = new TMultiGraph();
  TLegend* legSigma = new TLegend(0.55, 0.70, 0.88, 0.88);
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i)
      if (allFitResults[w][i].success)
        g->SetPoint(np++, allSliceCenters[w][i], allFitResults[w][i].sigma);
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.5);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgSigma->Add(g, "LP");
      legSigma->AddEntry(g, Form("%s", widthLabels[w].Data()), "lp");
    }
  }
  mgSigma->SetTitle("Pion Width vs. a;a-parameter;#sigma [MeV/c^{2}]");
  mgSigma->Draw("A");
  mgSigma->GetXaxis()->SetLimits(0.0, 12.0);
  mgSigma->GetYaxis()->SetRangeUser(0.0, 35.0);
  
  TLine* transLineSigma = new TLine(regularMaxA, 0, regularMaxA, 35);
  transLineSigma->SetLineColor(kMagenta);
  transLineSigma->SetLineStyle(kDashed);
  transLineSigma->Draw("same");
  
  legSigma->Draw();
  cPar->Modified();
  cPar->Update();

  // --- Save
  for (int w = 0; w < nWidths; ++w) {
    cFits[w]->SaveAs(Form("pion_fits_%s.png", widthLabels[w].Data()));
    cContours[w]->SaveAs(Form("pion_contours_%s.png", widthLabels[w].Data()));
  }
  cCombined3->SaveAs("pion_contours_combined_3sigma.png");
  cCombined1->SaveAs("pion_contours_combined_1sigma.png");
  cPBeta->SaveAs("pion_p_vs_beta.png");
  cPM2->SaveAs("pion_p_vs_mass2.png");
  cPar->SaveAs("pion_parameters.png");

  // Output file
  std::ofstream outfile("pion_fit_results.txt");
  outfile << "Width\ta-center\ta-low\ta-high\tWindowWidth\tMean\tSigma\tPolyOrder\tChi2NDF\tIsFinal\tStatus" << std::endl;
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i) {
      FitResult& r = allFitResults[w][i];
      outfile << widthLabels[w] << "\t" << allSliceCenters[w][i] << "\t" 
              << r.lowerBound << "\t" << r.upperBound << "\t" << r.windowWidth << "\t"
              << r.mean << "\t" << r.sigma << "\t" << r.polyOrder << "\t" << r.chi2ndf << "\t"
              << (r.isFinalSlice ? "YES" : "NO") << "\t"
              << (r.success ? "OK" : "FAIL") << std::endl;
    }
  }
  outfile.close();

  cout << "\n=== Analysis Complete ===" << endl;
  cout << "Regular fits: a = " << startLower << " to " << regularMaxA << endl;
  cout << "Final slice: a = " << regularMaxA << " to " << finalSliceEnd << " (width = " << (finalSliceEnd - regularMaxA) << ")" << endl;
  cout << "\nThis corresponds approximately to:" << endl;
  double pAtRegMax, betaAtRegMax, pAtFinal, betaAtFinal;
  if (massAToMomBeta(139.57, regularMaxA, sSquared, pAtRegMax, betaAtRegMax)) {
    cout << "  a = " << regularMaxA << " -> p ~ " << pAtRegMax << " MeV/c, beta ~ " << betaAtRegMax << endl;
  }
  if (massAToMomBeta(139.57, (regularMaxA + finalSliceEnd)/2, sSquared, pAtFinal, betaAtFinal)) {
    cout << "  Final slice center (a=" << (regularMaxA + finalSliceEnd)/2 << ") -> p ~ " << pAtFinal << " MeV/c" << endl;
  }
}
