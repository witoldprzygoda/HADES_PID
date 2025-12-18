// pid_macro_sliding_window.C
// Modified to use sliding/overlapping windows instead of discrete slices
// - Uses a moving window (e.g., 0.3-0.5, 0.31-0.51, 0.32-0.52, etc.)
// - Creates many more fit points for smoother contours
// - Only displays first 10 fits in canvas (to keep figures readable)
// - All fits are used for contours and parameter plots

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
#include "TAxis.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TChain.h"
#include "TSystem.h"

using std::cout; using std::endl;

//void pid_macro() {
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
  //const double sSquared = 1e-6;
  const double sSquared = 1e-4;

  // --- Build 2D hist: X=mass-like, Y=a-parameter (both signed by p)
  const char* h2name = "h2_pid_mass_vs_a";
  TString drawCmd = Form(
    "sqrt(1 + %.1e*pim_p*pim_p - pow(1-pim_beta*pim_beta,2)) * sign(pim_p) : "
    "pim_p*sqrt(1/pow(pim_beta,2) - 1) * sign(pim_p) >> %s(300,0,300,300,0,3)",
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

  // --- Define sliding window parameters
  const double windowWidth = 0.2;   // Width of each window (e.g., 0.5 - 0.3 = 0.2)
  const double stepSize = 0.01;     // Step between consecutive windows
  const double startLower = 0.3;    // Starting lower bound
  const double maxUpper = 2.3;      // Maximum upper bound we want to reach

  // Calculate number of slices
  const int nSlices = static_cast<int>((maxUpper - windowWidth - startLower) / stepSize) + 1;
  
  cout << "Using sliding windows: width=" << windowWidth 
       << ", step=" << stepSize 
       << ", total slices=" << nSlices << endl;

  // --- Storage (using vectors for dynamic size)
  std::vector<double> sliceCenters(nSlices, 0.0);
  std::vector<double> pionMean(nSlices, 0.0);
  std::vector<double> pionSigma(nSlices, 0.0);
  std::vector<double> pionAmplitude(nSlices, 0.0);
  std::vector<bool>   fitSuccess(nSlices, false);

  // --- Define which windows to display (corresponding to original discrete slices)
  // Original slices were: [0.3,0.5], [0.5,0.7], [0.7,0.9], [0.9,1.1], [1.1,1.3], 
  //                       [1.3,1.5], [1.5,1.7], [1.7,1.9], [1.9,2.1], [2.1,2.3]
  // These have lower bounds at: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1
  // With startLower=0.3 and stepSize=0.01: index = (lowerBound - 0.3) / 0.01
  std::vector<int> displayIndices;
  for (double lb = 0.3; lb <= 2.1 + 0.001; lb += 0.2) {
    int idx = static_cast<int>((lb - startLower) / stepSize + 0.5); // round to nearest
    if (idx >= 0 && idx < nSlices) {
      displayIndices.push_back(idx);
    }
  }
  
  cout << "Total slices to compute: " << nSlices << endl;
  cout << "Will display fits at indices: ";
  for (size_t d = 0; d < displayIndices.size(); ++d) {
    cout << displayIndices[d];
    if (d < displayIndices.size() - 1) cout << ", ";
  }
  cout << endl;
  
  // --- Canvas for fits (show representative windows matching original discrete slices)
  TCanvas* cFit = new TCanvas("c_fit", "Pion Fits - Original 10 slices: [0.3-0.5], [0.5-0.7], ..., [2.1-2.3]", 1200, 800);
  cFit->Divide(4, 3); // 12 pads, use first 10

  TCanvas* cSel = new TCanvas("c_sel", "Pion Selection", 800, 600);

  const double fitRangeMin = 100.0;
  const double fitRangeMax = 180.0;

  // --- Iterate over sliding windows
  for (int i = 0; i < nSlices; ++i) {
    const double lowerBound = startLower + i * stepSize;
    const double upperBound = lowerBound + windowWidth;
    sliceCenters[i] = 0.5 * (lowerBound + upperBound);

    // Check if this window should be displayed (matches original discrete slices)
    bool drawThisFit = false;
    int padNumber = 0;
    for (size_t d = 0; d < displayIndices.size(); ++d) {
      if (i == displayIndices[d]) {
        drawThisFit = true;
        padNumber = d + 1;
        cout << Form("Drawing fit %d (%.2f-%.2f) in pad %d", i, lowerBound, upperBound, padNumber) << endl;
        break;
      }
    }
    
    TPad* pad = nullptr;
    
    if (drawThisFit) {
      cFit->cd(padNumber);
      pad = static_cast<TPad*>(cFit->GetPad(padNumber));
      pad->SetLeftMargin(0.12);
      pad->SetRightMargin(0.04);
      pad->SetTopMargin(0.08);
      pad->SetBottomMargin(0.12);
    }

    // ProjectionX on X (mass) for Y in [lowerBound, upperBound]
    const int ybin_lo = h2->GetYaxis()->FindFixBin(lowerBound + 1e-6);
    const int ybin_hi = h2->GetYaxis()->FindFixBin(upperBound - 1e-6);
    TString projName = Form("proj_%d", i);
    if (gDirectory->FindObject(projName)) gDirectory->Delete(Form("%s;*", projName.Data()));
    TH1D* proj = h2->ProjectionX(projName, ybin_lo, ybin_hi, "e");

    // Defaults
    pionMean[i]      = 139.57;
    pionSigma[i]     = 10.0;
    pionAmplitude[i] = 0.0;
    fitSuccess[i]    = false;

    // Basic content checks
    const double entries = proj ? proj->GetEntries() : 0.0;
    const int bin_min = proj ? proj->FindFixBin(fitRangeMin) : 1;
    const int bin_max = proj ? proj->FindFixBin(fitRangeMax) : 0;
    const double integral_roi = (proj && bin_max >= bin_min) ? proj->Integral(bin_min, bin_max) : 0.0;

    if (i < 5 || i % 20 == 0) {  // Print only some to avoid spam
      cout << Form("Slice %3d: a in [%.2f, %.2f], entries=%.0f, integral[%.0f,%.0f]=%.0f",
                    i, lowerBound, upperBound, entries, fitRangeMin, fitRangeMax, integral_roi) << endl;
    }

    if (proj && entries > 20 && integral_roi > 5.0) {
      // Seed peak near the maximum inside ROI
      int maxBin = 0; double maxVal = -1.0;
      for (int b = bin_min; b <= bin_max; ++b) {
        double v = proj->GetBinContent(b);
        if (v > maxVal) { maxVal = v; maxBin = b; }
      }
      const double peakPos = (maxBin>0) ? proj->GetBinCenter(maxBin) : 139.57;

      TF1* fitFunc = new TF1(Form("fit_%d", i), "gaus(0) + pol1(3)", fitRangeMin, fitRangeMax);
      fitFunc->SetLineColor(kRed);
      // Init
      fitFunc->SetParameter(0, std::max(1.0, maxVal)); // amplitude
      fitFunc->SetParameter(1, peakPos);               // mean
      fitFunc->SetParameter(2, 10.0);                  // sigma
      fitFunc->SetParameter(3, proj->GetBinContent(bin_min)); // bkg level guess
      fitFunc->SetParameter(4, 0.0);                   // bkg slope
      // Limits
      fitFunc->SetParLimits(0, 0.0, std::max(10.0, 2.0*maxVal+1.0));
      fitFunc->SetParLimits(1, 120.0, 160.0);
      fitFunc->SetParLimits(2, 2.0, 30.0);

      // Quiet, return, use range
      TFitResultPtr result = proj->Fit(fitFunc, "SQR");

      bool valid = false; int status = -1;
      if (result.Get()) {
        status = result->Status();
        valid = result->IsValid() && status == 0;
      }

      if (valid) {
        pionMean[i]      = fitFunc->GetParameter(1);
        pionSigma[i]     = fitFunc->GetParameter(2);
        pionAmplitude[i] = fitFunc->GetParameter(0);
        fitSuccess[i]    = true;
      }
    }

    // Display cosmetics (only for shown fits)
    if (proj && drawThisFit) {
      // Draw the histogram first
      proj->GetXaxis()->SetRangeUser(fitRangeMin, fitRangeMax);
      proj->SetTitle(Form("%.2f < a < %.2f", lowerBound, upperBound));
      proj->GetXaxis()->SetTitle("Mass [MeV/c^{2}]");
      proj->GetYaxis()->SetTitle("Counts");
      proj->GetXaxis()->SetTitleSize(0.05);
      proj->GetYaxis()->SetTitleSize(0.05);
      proj->GetXaxis()->SetTitleOffset(1.0);
      proj->GetYaxis()->SetTitleOffset(1.2);
      proj->SetLineColor(kBlack);
      proj->Draw("E");

      // Draw fit components if fit succeeded
      if (fitSuccess[i]) {
        // Redraw the fit function
        TF1* fitFunc = (TF1*)gDirectory->Get(Form("fit_%d", i));
        if (fitFunc) fitFunc->Draw("same");
        
        // Draw signal component
        TF1* signalOnly = new TF1(Form("signal_%d", i), "gaus", fitRangeMin, fitRangeMax);
        signalOnly->SetParameters(pionAmplitude[i], pionMean[i], pionSigma[i]);
        signalOnly->SetLineColor(kBlue);
        signalOnly->SetLineStyle(kDashed);
        signalOnly->Draw("same");

        // Draw background component  
        TF1* fitFuncForBkg = (TF1*)gDirectory->Get(Form("fit_%d", i));
        if (fitFuncForBkg) {
          TF1* bkgOnly = new TF1(Form("bkg_%d", i), "pol1", fitRangeMin, fitRangeMax);
          bkgOnly->SetParameters(fitFuncForBkg->GetParameter(3), fitFuncForBkg->GetParameter(4));
          bkgOnly->SetLineColor(kGreen+1);
          bkgOnly->SetLineStyle(kDashed);
          bkgOnly->Draw("same");
        }
      }

      TLatex latex; latex.SetNDC(); latex.SetTextSize(0.05); latex.SetTextColor(kBlack);
      if (fitSuccess[i]) {
        latex.DrawLatex(0.15, 0.85, Form("Mean = %.1f", pionMean[i]));
        latex.DrawLatex(0.15, 0.78, Form("#sigma = %.1f", pionSigma[i]));
      } else {
        latex.DrawLatex(0.15, 0.85, "Fit failed / empty");
      }
      
      pad->Modified();
      pad->Update();
    }
    
    // Don't delete displayed projections - ROOT manages them
    // Only clean up non-displayed ones
    if (!drawThisFit && proj) {
      delete proj;
    }
  }

  cout << "Completed " << nSlices << " sliding window fits." << endl;
  
  // Final update of the fit canvas
  cFit->Modified();
  cFit->Update();

  // --- Selection contours on 2D
  cSel->cd();
  h2->Draw("colz");

  std::vector<double> massPoints[3], aPoints[3];
  
  // Forward pass (mean + n*sigma)
  for (int i = 0; i < nSlices; ++i) {
    if (fitSuccess[i]) {
      for (int s = 0; s < 3; ++s) {
        const double nS = s + 1;
        massPoints[s].push_back(pionMean[i] + nS * pionSigma[i]);
        aPoints[s].push_back(sliceCenters[i]);
      }
    }
  }
  
  // Backward pass (mean - n*sigma)
  for (int i = nSlices - 1; i >= 0; --i) {
    if (fitSuccess[i]) {
      for (int s = 0; s < 3; ++s) {
        const double nS = s + 1;
        massPoints[s].push_back(pionMean[i] - nS * pionSigma[i]);
        aPoints[s].push_back(sliceCenters[i]);
      }
    }
  }

  TGraph* contours[3] = {nullptr, nullptr, nullptr};
  int colors[3] = {kBlue, kGreen+2, kRed};
  for (int s = 0; s < 3; ++s) {
    if (!massPoints[s].empty()) {
      contours[s] = new TGraph((int)massPoints[s].size(), massPoints[s].data(), aPoints[s].data());
      contours[s]->SetLineColor(colors[s]);
      contours[s]->SetLineWidth(2);
      contours[s]->Draw("L");
    }
  }

  if (contours[0] || contours[1] || contours[2]) {
    TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
    if (contours[0]) leg->AddEntry(contours[0], "1#sigma", "l");
    if (contours[1]) leg->AddEntry(contours[1], "2#sigma", "l");
    if (contours[2]) leg->AddEntry(contours[2], "3#sigma", "l");
    leg->SetBorderSize(1);
    leg->Draw();
  }

  // --- Mean/Sigma vs a
  TCanvas* cPar = new TCanvas("c_par", "Pion Parameters vs. a", 1000, 450);
  cPar->Divide(2,1);

  // Mean
  cPar->cd(1);
  TGraph* gMean = new TGraph();
  int nPoints = 0;
  for (int i = 0; i < nSlices; ++i) {
    if (fitSuccess[i]) {
      gMean->SetPoint(nPoints++, sliceCenters[i], pionMean[i]);
    }
  }
  gMean->SetTitle("Pion Mass vs. a-parameter");
  gMean->GetXaxis()->SetTitle("a-parameter");
  gMean->GetYaxis()->SetTitle("Mean [MeV/c^{2}]");
  gMean->GetYaxis()->SetRangeUser(130, 150);
  gMean->SetMarkerStyle(20);
  gMean->SetMarkerSize(0.5);  // Smaller markers since we have many points
  gMean->SetMarkerColor(kBlue);
  gMean->SetLineColor(kBlue);
  gMean->Draw("ALP");
  {
    TF1* constLine = new TF1("constLine", "139.57", 0, 3);
    constLine->SetLineColor(kRed);
    constLine->SetLineStyle(kDashed);
    constLine->Draw("same");
    TLegend* legMean = new TLegend(0.7, 0.2, 0.9, 0.4);
    legMean->AddEntry(gMean, "Fitted Mean", "lp");
    legMean->AddEntry(constLine, "Pion Mass (139.57)", "l");
    legMean->Draw();
  }

  // Sigma
  cPar->cd(2);
  TGraph* gSigma = new TGraph(); 
  nPoints = 0;
  for (int i = 0; i < nSlices; ++i) {
    if (fitSuccess[i]) {
      gSigma->SetPoint(nPoints++, sliceCenters[i], pionSigma[i]);
    }
  }
  gSigma->SetTitle("Pion Width vs. a-parameter");
  gSigma->GetXaxis()->SetTitle("a-parameter");
  gSigma->GetYaxis()->SetTitle("#sigma [MeV/c^{2}]");
  gSigma->SetMarkerStyle(20);
  gSigma->SetMarkerSize(0.5);  // Smaller markers since we have many points
  gSigma->SetMarkerColor(kRed);
  gSigma->SetLineColor(kRed);
  gSigma->Draw("ALP");

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
    TLegend* legFull = new TLegend(0.7, 0.7, 0.9, 0.9);
    legFull->AddEntry(hfull, "Mass Spectrum", "lf");
    legFull->AddEntry(pionLine, "Pion Mass (139.57)", "l");
    legFull->Draw();
  }

  // --- Save results
  std::ofstream outfile("pion_fit_results.txt");
  outfile << "a-parameter\tMean [MeV/c^2]\tSigma [MeV/c^2]\tAmplitude\tFit Status" << std::endl;
  cout << "\nSummary of fit results (first 20 and last 5):" << std::endl;
  cout << "a-parameter\tMean [MeV/c^2]\tSigma [MeV/c^2]\tAmplitude\tFit Status" << std::endl;
  
  for (int i = 0; i < nSlices; ++i) {
    outfile << sliceCenters[i] << "\t"
            << pionMean[i] << "\t"
            << pionSigma[i] << "\t"
            << pionAmplitude[i] << "\t"
            << (fitSuccess[i] ? "Success" : "Failed") << std::endl;
    
    // Print only subset to console to avoid spam
    if (i < 20 || i >= nSlices - 5) {
      cout << sliceCenters[i] << "\t"
           << pionMean[i] << "\t"
           << pionSigma[i] << "\t"
           << pionAmplitude[i] << "\t"
           << (fitSuccess[i] ? "Success" : "Failed") << std::endl;
    } else if (i == 20) {
      cout << "... (" << (nSlices - 25) << " more entries) ..." << std::endl;
    }
  }
  outfile.close();

  // Count successful fits
  int nSuccess = 0;
  for (int i = 0; i < nSlices; ++i) {
    if (fitSuccess[i]) nSuccess++;
  }
  cout << "\nTotal successful fits: " << nSuccess << " out of " << nSlices << endl;

  // --- Save canvases
  cFit->SaveAs("pion_fits.png");
  cSel->SaveAs("pion_contours.png");
  cPar->SaveAs("pion_parameters.png");
  cMass->SaveAs("pion_mass_spectrum.png");

  cout << "\nAnalysis complete. Results saved to:" << endl
       << "- pion_fits.png (shows 10 representative windows matching original slices)" << endl
       << "- pion_contours.png (smooth contours from all fits)" << endl
       << "- pion_parameters.png (parameters from all fits)" << endl
       << "- pion_mass_spectrum.png" << endl
       << "- pion_fit_results.txt (all " << nSlices << " fits)" << endl;
}
