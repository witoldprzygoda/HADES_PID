// pid_macro_sliding_window.C
// Performs 4 parallel sliding window scans with different widths:
// 1x (0.2), 2x (0.4), 4x (0.8), 8x (1.6)
// ITERATIVE BACKGROUND: tries pol0, pol1, pol2, pol3, pol4 and picks best chi2/ndf
// All scans end at the same a-value (centers aligned at ~5.9)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

// Structure to hold fit results
struct FitResult {
  bool success;
  double mean;
  double sigma;
  double amplitude;
  int polyOrder;        // Which polynomial order was best (0-4)
  double chi2ndf;
  std::vector<double> bkgParams;  // Background parameters
  double entries;
};

// Function to try fitting with different polynomial backgrounds
FitResult tryFitWithPolynomials(TH1D* proj, double fitMin, double fitMax, int sliceIdx, int widthIdx) {
  FitResult best;
  best.success = false;
  best.mean = 139.57;
  best.sigma = 10.0;
  best.amplitude = 0.0;
  best.polyOrder = 1;
  best.chi2ndf = 1e9;
  best.entries = proj ? proj->GetEntries() : 0;

  if (!proj || best.entries < 20) return best;

  int bin_min = proj->FindFixBin(fitMin);
  int bin_max = proj->FindFixBin(fitMax);
  double integral = proj->Integral(bin_min, bin_max);
  if (integral < 5.0) return best;

  // Find peak position
  int maxBin = 0;
  double maxVal = -1.0;
  for (int b = bin_min; b <= bin_max; ++b) {
    double v = proj->GetBinContent(b);
    if (v > maxVal) { maxVal = v; maxBin = b; }
  }
  double peakPos = (maxBin > 0) ? proj->GetBinCenter(maxBin) : 139.57;

  // Try polynomial orders 0 through 4
  for (int polyOrder = 0; polyOrder <= 4; ++polyOrder) {
    // Build fit function: gaus(0) + pol<N>(3)
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
    
    // Initialize Gaussian parameters
    fitFunc->SetParameter(0, std::max(1.0, maxVal * 0.8));  // amplitude
    fitFunc->SetParameter(1, peakPos);                       // mean
    fitFunc->SetParameter(2, 10.0);                          // sigma
    
    // Initialize background parameters
    double bkgLevel = proj->GetBinContent(bin_min);
    fitFunc->SetParameter(3, bkgLevel);
    for (int p = 1; p < nBkgParams; ++p) {
      fitFunc->SetParameter(3 + p, 0.0);
    }

    // Set parameter limits
    fitFunc->SetParLimits(0, 0.0, std::max(10.0, 2.0 * maxVal + 1.0));
    fitFunc->SetParLimits(1, 100.0, 180.0);
    fitFunc->SetParLimits(2, 2.0, 40.0);

    // Perform fit
    TFitResultPtr result = proj->Fit(fitFunc, "SQR0");  // 0 = don't draw

    bool valid = false;
    double chi2ndf = 1e9;

    if (result.Get() && result->IsValid() && result->Status() == 0) {
      double fMean = fitFunc->GetParameter(1);
      double fSigma = fitFunc->GetParameter(2);
      double fAmp = fitFunc->GetParameter(0);

      // Quality checks
      if (fMean > 115.0 && fMean < 165.0 && 
          fSigma > 2.0 && fSigma < 40.0 && 
          fAmp > 0.5) {
        valid = true;
        if (result->Ndf() > 0) {
          chi2ndf = result->Chi2() / result->Ndf();
        } else {
          chi2ndf = 1e6;  // Penalize if no DOF
        }
      }
    }

    // Check if this is the best fit so far
    if (valid && chi2ndf < best.chi2ndf) {
      best.success = true;
      best.mean = fitFunc->GetParameter(1);
      best.sigma = fitFunc->GetParameter(2);
      best.amplitude = fitFunc->GetParameter(0);
      best.polyOrder = polyOrder;
      best.chi2ndf = chi2ndf;
      best.bkgParams.clear();
      for (int p = 0; p < nBkgParams; ++p) {
        best.bkgParams.push_back(fitFunc->GetParameter(3 + p));
      }
    }

    delete fitFunc;
  }

  return best;
}

// Build background function string based on polynomial order
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

  // --- Build a TChain from a hard-wired file list
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
      cout << "[WARN] missing file: " << fn << " (skipping)" << endl;
      continue;
    }
    chain->Add(fn);
    ++added;
  }
  Long64_t nEnt = chain->GetEntries();
  if (added == 0 || nEnt <= 0) {
    cout << "No entries in TChain for the hard-wired file list." << endl;
    return;
  }
  cout << "TChain built: files added = " << added << ", entries = " << nEnt << endl;

  // --- Parameter
  const double sSquared = 1e-4;

  // --- Build 2D hist: X=mass-like, Y=a-parameter
  const char* h2name = "h2_pid_mass_vs_a";
  TString drawCmd = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) * sign(pim_p) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) * sign(pim_p) >> %s(300,0,300,700,0,7)",
    sSquared, h2name);

  if (gDirectory->FindObject(h2name)) gDirectory->Delete(Form("%s;*", h2name));
  chain->Draw(drawCmd, "isBest==1", "colz");
  TH2F* h2 = static_cast<TH2F*>(gDirectory->Get(h2name));
  if (!h2) {
    cout << "Failed to create histogram" << endl;
    return;
  }

  h2->SetTitle(Form("Mass vs a-parameter (s^{2} = %.1e)", sSquared));
  h2->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
  h2->GetYaxis()->SetTitle("a * sign(p)");

  // --- Sliding window parameters
  const double baseWidth = 0.2;
  const double stepSize = 0.01;
  const double startLower = 0.3;
  const double targetMaxCenter = 5.9;
  
  // Fit range extended to 80-220
  const double fitRangeMin = 80.0;
  const double fitRangeMax = 220.0;

  // Width multipliers for 4 parallel scans
  const int nWidths = 4;
  double widthMultipliers[nWidths] = {1.0, 2.0, 4.0, 8.0};
  TString widthLabels[nWidths] = {"1x", "2x", "4x", "8x"};
  int widthColors[nWidths] = {kBlue, kGreen+2, kOrange+1, kRed};

  // --- Storage for all width scans
  std::vector<std::vector<double>> allSliceCenters(nWidths);
  std::vector<std::vector<FitResult>> allFitResults(nWidths);

  // --- Create canvases for fit displays (one per width)
  TCanvas* cFits[nWidths];
  const int nDisplayPerWidth = 12;
  
  for (int w = 0; w < nWidths; ++w) {
    TString canvasName = Form("c_fit_%s", widthLabels[w].Data());
    TString canvasTitle = Form("Pion Fits - Width %s (%.2f)", widthLabels[w].Data(), baseWidth * widthMultipliers[w]);
    cFits[w] = new TCanvas(canvasName, canvasTitle, 1200, 800);
    cFits[w]->Divide(4, 3);
    cout << "Created canvas: " << canvasName << endl;
  }

  // --- Perform scans for each width
  for (int w = 0; w < nWidths; ++w) {
    double currentWidth = baseWidth * widthMultipliers[w];
    double maxUpper = targetMaxCenter + currentWidth / 2.0;
    
    int nSlices = static_cast<int>((maxUpper - currentWidth - startLower) / stepSize) + 1;
    if (nSlices < 1) nSlices = 1;
    
    cout << "\n========================================" << endl;
    cout << "=== Width " << widthLabels[w] << " (" << currentWidth << ") ===" << endl;
    cout << "========================================" << endl;
    cout << "maxUpper = " << maxUpper << ", nSlices = " << nSlices << endl;
    
    double firstCenter = startLower + currentWidth / 2.0;
    double lastCenter = startLower + (nSlices - 1) * stepSize + currentWidth / 2.0;
    cout << "Centers: first=" << firstCenter << ", last=" << lastCenter << endl;

    // Resize storage
    allSliceCenters[w].resize(nSlices);
    allFitResults[w].resize(nSlices);

    // Determine display indices - evenly spaced
    std::vector<int> displayIndices;
    if (nSlices >= nDisplayPerWidth) {
      for (int d = 0; d < nDisplayPerWidth; ++d) {
        int idx = (int)((double)d * (nSlices - 1) / (nDisplayPerWidth - 1) + 0.5);
        displayIndices.push_back(idx);
      }
    } else {
      for (int i = 0; i < nSlices; ++i) displayIndices.push_back(i);
    }
    
    cout << "Display indices (" << displayIndices.size() << "): ";
    for (size_t d = 0; d < displayIndices.size(); ++d) cout << displayIndices[d] << " ";
    cout << endl;

    // --- First pass: perform all fits
    int nSuccess = 0;
    int polyOrderCounts[5] = {0, 0, 0, 0, 0};
    
    for (int i = 0; i < nSlices; ++i) {
      double lowerBound = startLower + i * stepSize;
      double upperBound = lowerBound + currentWidth;
      double center = 0.5 * (lowerBound + upperBound);
      allSliceCenters[w][i] = center;

      // Create projection
      int ybin_lo = h2->GetYaxis()->FindFixBin(lowerBound + 1e-6);
      int ybin_hi = h2->GetYaxis()->FindFixBin(upperBound - 1e-6);
      
      TString projName = Form("proj_fit_w%d_%d", w, i);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2->ProjectionX(projName, ybin_lo, ybin_hi, "e");

      // Try fitting with different polynomial backgrounds
      FitResult result = tryFitWithPolynomials(proj, fitRangeMin, fitRangeMax, i, w);
      allFitResults[w][i] = result;

      if (result.success) {
        nSuccess++;
        polyOrderCounts[result.polyOrder]++;
      }

      // Progress output
      if (i < 3 || i % 100 == 0 || i == nSlices - 1) {
        cout << Form("  Slice %4d/%d: a=%.2f, entries=%.0f, fit=%s",
                     i, nSlices, center, result.entries,
                     result.success ? Form("OK (pol%d, chi2/ndf=%.2f)", result.polyOrder, result.chi2ndf) : "FAIL") << endl;
      }

      delete proj;
    }

    cout << "\nWidth " << widthLabels[w] << " summary: " << nSuccess << "/" << nSlices << " successful" << endl;
    cout << "Polynomial order distribution: ";
    for (int p = 0; p <= 4; ++p) cout << "pol" << p << "=" << polyOrderCounts[p] << " ";
    cout << endl;

    // --- Second pass: create display histograms
    cout << "\nCreating display histograms for width " << widthLabels[w] << "..." << endl;
    
    for (size_t d = 0; d < displayIndices.size(); ++d) {
      int i = displayIndices[d];
      int padNum = d + 1;
      
      double lowerBound = startLower + i * stepSize;
      double upperBound = lowerBound + currentWidth;
      double center = allSliceCenters[w][i];
      FitResult& result = allFitResults[w][i];

      cFits[w]->cd(padNum);
      gPad->SetLeftMargin(0.12);
      gPad->SetRightMargin(0.04);
      gPad->SetTopMargin(0.10);
      gPad->SetBottomMargin(0.12);

      // Create projection for display
      int ybin_lo = h2->GetYaxis()->FindFixBin(lowerBound + 1e-6);
      int ybin_hi = h2->GetYaxis()->FindFixBin(upperBound - 1e-6);
      
      TString projName = Form("proj_disp_w%d_d%d", w, (int)d);
      if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
      TH1D* proj = h2->ProjectionX(projName, ybin_lo, ybin_hi, "e");

      if (!proj) {
        cout << "  WARNING: Could not create projection for display " << d << endl;
        continue;
      }

      proj->SetTitle(Form("%.2f < a < %.2f", lowerBound, upperBound));
      proj->GetXaxis()->SetRangeUser(fitRangeMin, fitRangeMax);
      proj->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
      proj->GetYaxis()->SetTitle("Counts");
      proj->GetXaxis()->SetTitleSize(0.05);
      proj->GetYaxis()->SetTitleSize(0.05);
      proj->SetLineColor(kBlack);
      proj->SetMarkerStyle(20);
      proj->SetMarkerSize(0.4);
      proj->Draw("E");

      // Draw fit if successful
      if (result.success) {
        // Build and draw total fit function
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
        for (size_t p = 0; p < result.bkgParams.size(); ++p) {
          fitFunc->SetParameter(3 + p, result.bkgParams[p]);
        }
        fitFunc->SetLineColor(kRed);
        fitFunc->SetLineWidth(2);
        fitFunc->Draw("same");

        // Signal component (Gaussian)
        TF1* sig = new TF1(Form("sig_w%d_d%d", w, (int)d), "gaus", fitRangeMin, fitRangeMax);
        sig->SetParameters(result.amplitude, result.mean, result.sigma);
        sig->SetLineColor(kBlue);
        sig->SetLineStyle(kDashed);
        sig->SetLineWidth(2);
        sig->Draw("same");

        // Background component
        TString bkgExpr = getBkgFuncExpr(result.polyOrder);
        TF1* bkg = new TF1(Form("bkg_w%d_d%d", w, (int)d), bkgExpr, fitRangeMin, fitRangeMax);
        for (size_t p = 0; p < result.bkgParams.size(); ++p) {
          bkg->SetParameter(p, result.bkgParams[p]);
        }
        bkg->SetLineColor(kGreen+2);
        bkg->SetLineStyle(kDashed);
        bkg->SetLineWidth(2);
        bkg->Draw("same");
      }

      // Labels
      TLatex tex;
      tex.SetNDC();
      tex.SetTextSize(0.042);
      if (result.success) {
        tex.DrawLatex(0.52, 0.82, Form("Mean=%.1f", result.mean));
        tex.DrawLatex(0.52, 0.74, Form("#sigma=%.1f", result.sigma));
        tex.DrawLatex(0.52, 0.66, Form("Bkg: pol%d", result.polyOrder));
        tex.DrawLatex(0.52, 0.58, Form("#chi^{2}/ndf=%.1f", result.chi2ndf));
      } else {
        tex.SetTextColor(kRed);
        tex.DrawLatex(0.52, 0.82, "Fit failed");
        tex.SetTextColor(kBlack);
      }
      tex.DrawLatex(0.52, 0.50, Form("N=%.0f", result.entries));

      gPad->Modified();
      gPad->Update();
      
      cout << "  Pad " << padNum << ": slice " << i << ", a=" << center << ", fit=" << (result.success ? "OK" : "FAIL") << endl;
    }

    cFits[w]->Modified();
    cFits[w]->Update();
    cout << "Canvas " << widthLabels[w] << " updated." << endl;
  }

  // --- Create 2D contour plots for each width
  TCanvas* cContours[nWidths];
  
  for (int w = 0; w < nWidths; ++w) {
    double currentWidth = baseWidth * widthMultipliers[w];
    cContours[w] = new TCanvas(Form("c_contour_%s", widthLabels[w].Data()),
                               Form("Pion Selection - Width %s (%.2f)", widthLabels[w].Data(), currentWidth),
                               800, 600);
    cContours[w]->cd();
    h2->Draw("colz");

    // Build contours
    std::vector<double> massPoints[3], aPoints[3];
    int nSlices = allSliceCenters[w].size();
    
    // Forward pass
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          double nSig = s + 1;
          massPoints[s].push_back(allFitResults[w][i].mean + nSig * allFitResults[w][i].sigma);
          aPoints[s].push_back(allSliceCenters[w][i]);
        }
      }
    }
    
    // Backward pass
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        for (int s = 0; s < 3; ++s) {
          double nSig = s + 1;
          massPoints[s].push_back(allFitResults[w][i].mean - nSig * allFitResults[w][i].sigma);
          aPoints[s].push_back(allSliceCenters[w][i]);
        }
      }
    }

    // Draw contours - thick lines
    int contColors[3] = {kBlue, kGreen+2, kRed};
    TGraph* graphs[3] = {nullptr, nullptr, nullptr};
    
    for (int s = 0; s < 3; ++s) {
      if (!massPoints[s].empty()) {
        graphs[s] = new TGraph(massPoints[s].size(), massPoints[s].data(), aPoints[s].data());
        graphs[s]->SetLineColor(contColors[s]);
        graphs[s]->SetLineWidth(6);
        graphs[s]->Draw("L");
      }
    }

    TLegend* leg = new TLegend(0.7, 0.7, 0.88, 0.88);
    leg->SetHeader(Form("Width: %s", widthLabels[w].Data()));
    if (graphs[0]) leg->AddEntry(graphs[0], "1#sigma", "l");
    if (graphs[1]) leg->AddEntry(graphs[1], "2#sigma", "l");
    if (graphs[2]) leg->AddEntry(graphs[2], "3#sigma", "l");
    leg->SetBorderSize(1);
    leg->Draw();

    cContours[w]->Modified();
    cContours[w]->Update();
  }

  // --- Combined contour comparison plot
  TCanvas* cCombined = new TCanvas("c_combined", "Contour Comparison - All Widths (3#sigma)", 1000, 800);
  cCombined->cd();
  h2->Draw("colz");

  TLegend* legComb = new TLegend(0.7, 0.65, 0.88, 0.88);
  legComb->SetHeader("3#sigma contours");

  for (int w = 0; w < nWidths; ++w) {
    std::vector<double> massPoints, aPoints;
    int nSlices = allSliceCenters[w].size();
    
    for (int i = 0; i < nSlices; ++i) {
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean + 3.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    }
    for (int i = nSlices - 1; i >= 0; --i) {
      if (allFitResults[w][i].success) {
        massPoints.push_back(allFitResults[w][i].mean - 3.0 * allFitResults[w][i].sigma);
        aPoints.push_back(allSliceCenters[w][i]);
      }
    }

    if (!massPoints.empty()) {
      TGraph* g = new TGraph(massPoints.size(), massPoints.data(), aPoints.data());
      g->SetLineColor(widthColors[w]);
      g->SetLineWidth(6);
      g->Draw("L");
      legComb->AddEntry(g, Form("Width %s", widthLabels[w].Data()), "l");
    }
  }
  legComb->Draw();
  cCombined->Modified();
  cCombined->Update();

  // --- Mean/Sigma vs a
  TCanvas* cPar = new TCanvas("c_par", "Pion Parameters vs. a", 1000, 450);
  cPar->Divide(2, 1);

  // Mean plot
  cPar->cd(1);
  gPad->SetLeftMargin(0.12);
  gPad->SetBottomMargin(0.12);
  
  TMultiGraph* mgMean = new TMultiGraph();
  TLegend* legMean = new TLegend(0.65, 0.70, 0.88, 0.88);
  
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allSliceCenters[w][i], allFitResults[w][i].mean);
      }
    }
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.4);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgMean->Add(g, "LP");
      legMean->AddEntry(g, Form("Width %s", widthLabels[w].Data()), "lp");
    }
  }
  
  mgMean->SetTitle("Pion Mass vs. a-parameter;a-parameter;Mean [MeV/c^{2}]");
  mgMean->Draw("A");
  mgMean->GetXaxis()->SetLimits(0.0, 6.0);
  mgMean->GetYaxis()->SetRangeUser(120.0, 160.0);
  
  TF1* constLine = new TF1("constLine", "139.57", 0, 6.5);
  constLine->SetLineColor(kBlack);
  constLine->SetLineStyle(kDashed);
  constLine->SetLineWidth(2);
  constLine->Draw("same");
  legMean->AddEntry(constLine, "m_{#pi} = 139.57", "l");
  legMean->Draw();

  // Sigma plot
  cPar->cd(2);
  gPad->SetLeftMargin(0.12);
  gPad->SetBottomMargin(0.12);
  
  TMultiGraph* mgSigma = new TMultiGraph();
  TLegend* legSigma = new TLegend(0.65, 0.70, 0.88, 0.88);
  
  for (int w = 0; w < nWidths; ++w) {
    TGraph* g = new TGraph();
    int np = 0;
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i) {
      if (allFitResults[w][i].success) {
        g->SetPoint(np++, allSliceCenters[w][i], allFitResults[w][i].sigma);
      }
    }
    if (np > 0) {
      g->SetMarkerStyle(20 + w);
      g->SetMarkerSize(0.4);
      g->SetMarkerColor(widthColors[w]);
      g->SetLineColor(widthColors[w]);
      mgSigma->Add(g, "LP");
      legSigma->AddEntry(g, Form("Width %s", widthLabels[w].Data()), "lp");
    }
  }
  
  mgSigma->SetTitle("Pion Width vs. a-parameter;a-parameter;#sigma [MeV/c^{2}]");
  mgSigma->Draw("A");
  mgSigma->GetXaxis()->SetLimits(0.0, 6.0);
  mgSigma->GetYaxis()->SetRangeUser(0.0, 35.0);
  legSigma->Draw();

  cPar->Modified();
  cPar->Update();

  // --- Full mass spectrum
  TCanvas* cMass = new TCanvas("c_mass", "Full Mass Spectrum", 800, 600);
  const char* hfullName = "hfull_mass";
  if (gDirectory->FindObject(hfullName)) gDirectory->Delete(Form("%s;*", hfullName));
  chain->Draw(Form("pim_p*sqrt(1/pow(pim_beta,2) - 1) >> %s(300,0,300)", hfullName), "isBest==1");
  TH1F* hfull = static_cast<TH1F*>(gDirectory->Get(hfullName));
  if (hfull) {
    hfull->SetTitle("Full Mass Spectrum");
    hfull->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
    hfull->GetYaxis()->SetTitle("Counts");
    hfull->SetLineColor(kBlue);
    hfull->SetFillColor(kCyan-10);
    TLine* pionLine = new TLine(139.57, 0, 139.57, hfull->GetMaximum());
    pionLine->SetLineColor(kRed);
    pionLine->SetLineWidth(2);
    pionLine->SetLineStyle(kDashed);
    pionLine->Draw("same");
  }

  // --- Save results
  std::ofstream outfile("pion_fit_results.txt");
  outfile << "Width\ta-center\tMean\tSigma\tAmplitude\tPolyOrder\tChi2/NDF\tStatus" << std::endl;
  
  for (int w = 0; w < nWidths; ++w) {
    for (size_t i = 0; i < allSliceCenters[w].size(); ++i) {
      FitResult& r = allFitResults[w][i];
      outfile << widthLabels[w] << "\t"
              << allSliceCenters[w][i] << "\t"
              << r.mean << "\t"
              << r.sigma << "\t"
              << r.amplitude << "\t"
              << r.polyOrder << "\t"
              << r.chi2ndf << "\t"
              << (r.success ? "OK" : "FAIL") << std::endl;
    }
  }
  outfile.close();

  // --- Save canvases
  for (int w = 0; w < nWidths; ++w) {
    cFits[w]->SaveAs(Form("pion_fits_%s.png", widthLabels[w].Data()));
    cContours[w]->SaveAs(Form("pion_contours_%s.png", widthLabels[w].Data()));
  }
  cCombined->SaveAs("pion_contours_combined.png");
  cPar->SaveAs("pion_parameters.png");
  cMass->SaveAs("pion_mass_spectrum.png");

  cout << "\n=== Analysis complete ===" << endl;
  cout << "Fit range: " << fitRangeMin << " - " << fitRangeMax << " MeV/c^2" << endl;
  cout << "Background: iterative pol0-pol4, best chi2/ndf selected" << endl;
  cout << "\nSaved files:" << endl;
  for (int w = 0; w < nWidths; ++w) {
    cout << "  - pion_fits_" << widthLabels[w] << ".png" << endl;
    cout << "  - pion_contours_" << widthLabels[w] << ".png" << endl;
  }
  cout << "  - pion_contours_combined.png" << endl;
  cout << "  - pion_parameters.png" << endl;
  cout << "  - pion_mass_spectrum.png" << endl;
  cout << "  - pion_fit_results.txt" << endl;
}
