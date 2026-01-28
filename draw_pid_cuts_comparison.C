// =====================================================
// draw_pid_cuts_comparison.C
// =====================================================
// 
// PURPOSE: Draw TCutG contours comparing experimental and simulation
//          PID cuts for protons, π+, π-, e+, and e-
//
// DRAWS: 1σ, 3σ, 5σ contours
//        - Experimental: solid lines
//        - Simulation: dashed lines
//
// COLOR SCHEME:
//        - Protons (p): Red family
//        - Pions+ (pip): Blue family  
//        - Pions- (pim): Green family
//        - Positrons (ep): Magenta family
//        - Electrons (em): Cyan family
//
// OUTPUT: 5 separate canvases (one per particle) + 1 summary canvas
//
// USAGE:
//   root -l 'draw_pid_cuts_comparison.C("pp45_pid_cuts_exp.root", "pp45_pid_cuts_sim.root")'
//
// =====================================================

#include <iostream>
#include <vector>

#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TCutG.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TF1.h"
#include "TLine.h"
#include "TAxis.h"

using std::cout; using std::endl;

// =====================================================
// HELPER: Load TCutG from file with unique clone name
// Tries multiple naming conventions for compatibility
// =====================================================
TCutG* loadCut(TFile* f, const TString& cutName, const TString& cloneName) {
  if (!f || f->IsZombie()) return nullptr;
  
  // Try different naming conventions
  std::vector<TString> tryNames;
  
  // Original format: prefix_cut_Xsig (e.g., p_cut_3sig)
  tryNames.push_back(cutName);
  
  // TCutG directory format: TCutG/cut_prefix_Xsig (e.g., TCutG/cut_ep_3sig)
  TString altName = cutName;
  altName.ReplaceAll("_cut_", "_");
  tryNames.push_back(Form("TCutG/cut_%s", altName.Data()));
  
  // Alternative: just cut_prefix_Xsig without directory
  tryNames.push_back(Form("cut_%s", altName.Data()));
  
  TCutG* cut = nullptr;
  for (const auto& name : tryNames) {
    cut = (TCutG*)f->Get(name);
    if (cut) break;
  }
  
  if (cut) {
    cut = (TCutG*)cut->Clone(cloneName);
  }
  return cut;
}

// =====================================================
// CONFIGURATION STRUCTURES
// =====================================================

struct ParticleConfig {
  TString prefix;      // Cut name prefix (p, pip, pim, ep, em)
  TString label;       // Display label (ROOT formatted)
  TString labelLeg;    // Label for legend
  double mass;         // Particle mass [MeV/c²]
  double pMax;         // Maximum momentum for display
  int color1sig;       // Color for 1σ
  int color3sig;       // Color for 3σ
  int color5sig;       // Color for 5σ
};

struct SigmaConfig {
  TString suffix;      // Cut name suffix (1sig, 3sig, 5sig)
  TString label;       // Display label
  int sigmaIdx;        // Index for color selection (0=1sig, 1=3sig, 2=5sig)
  int lineWidth;       // Line width
};

// =====================================================
// DRAW SINGLE PARTICLE CANVAS
// =====================================================
TCanvas* drawParticleCanvas(
    TFile* fExp, 
    TFile* fSim,
    const ParticleConfig& part,
    const std::vector<SigmaConfig>& sigmas,
    double betaMin,
    double betaMax
) {
  TString canvasName = Form("c_%s_comparison", part.prefix.Data());
  TString canvasTitle = Form("%s PID Cuts: Exp vs Sim", part.labelLeg.Data());
  
  TCanvas* canvas = new TCanvas(canvasName, canvasTitle, 900, 700);
  gPad->SetLeftMargin(0.14);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.08);
  gPad->SetBottomMargin(0.14);
  gPad->SetTickx(1);
  gPad->SetTicky(1);
  
  // Create frame
  TH1F* frame = gPad->DrawFrame(0, betaMin, part.pMax, betaMax);
  
  // Set axis titles
  frame->GetXaxis()->SetTitle("p [MeV/c]");
  frame->GetYaxis()->SetTitle("#beta");
  frame->GetXaxis()->SetTitleSize(0.055);
  frame->GetYaxis()->SetTitleSize(0.055);
  frame->GetXaxis()->SetLabelSize(0.045);
  frame->GetYaxis()->SetLabelSize(0.045);
  frame->GetXaxis()->SetTitleOffset(1.1);
  frame->GetYaxis()->SetTitleOffset(1.1);
  frame->GetXaxis()->SetNdivisions(510);
  frame->GetYaxis()->SetNdivisions(510);
  
  // Draw theoretical β(p) curve
  TF1* fTheory = new TF1(Form("fTheory_%s", part.prefix.Data()), 
                         "x/sqrt(x*x + [0]*[0])", 0, part.pMax);
  fTheory->SetParameter(0, part.mass);
  fTheory->SetLineColor(kGray+1);
  fTheory->SetLineStyle(2);
  fTheory->SetLineWidth(2);
  fTheory->Draw("same");
  
  // Create legend
  TLegend* leg = new TLegend(0.55, 0.15, 0.93, 0.50);
  leg->SetTextSize(0.035);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetHeader(Form("%s cuts", part.label.Data()));
  
  int nLoaded = 0;
  
  // Loop over sigma levels
  for (size_t s = 0; s < sigmas.size(); ++s) {
    const SigmaConfig& sig = sigmas[s];
    
    // Construct cut name
    TString cutName = Form("%s_cut_%s", part.prefix.Data(), sig.suffix.Data());
    
    // Get color based on sigma index
    int color;
    if (sig.sigmaIdx == 0) color = part.color1sig;
    else if (sig.sigmaIdx == 1) color = part.color3sig;
    else color = part.color5sig;
    
    // Create unique clone names
    TString expCloneName = Form("%s_exp_%s_single", part.prefix.Data(), sig.suffix.Data());
    TString simCloneName = Form("%s_sim_%s_single", part.prefix.Data(), sig.suffix.Data());
    
    // Load and draw experimental cut
    TCutG* cutExp = loadCut(fExp, cutName, expCloneName);
    if (cutExp) {
      cutExp->SetLineColor(color);
      cutExp->SetLineWidth(sig.lineWidth);
      cutExp->SetLineStyle(1);  // Solid for exp
      cutExp->SetFillStyle(0);
      cutExp->Draw("L same");
      leg->AddEntry(cutExp, Form("Exp %s", sig.label.Data()), "l");
      nLoaded++;
    } else {
      cout << "  [WARN] Missing exp cut: " << cutName << endl;
    }
    
    // Load and draw simulation cut
    TCutG* cutSim = loadCut(fSim, cutName, simCloneName);
    if (cutSim) {
      cutSim->SetLineColor(color);
      cutSim->SetLineWidth(sig.lineWidth);
      cutSim->SetLineStyle(2);  // Dashed for sim
      cutSim->SetFillStyle(0);
      cutSim->Draw("L same");
      leg->AddEntry(cutSim, Form("Sim %s", sig.label.Data()), "l");
      nLoaded++;
    } else {
      cout << "  [WARN] Missing sim cut: " << cutName << endl;
    }
  }
  
  // Add theory line to legend
  leg->AddEntry(fTheory, "#beta_{theory}(p)", "l");
  leg->Draw();
  
  // Add particle label
  TLatex tex;
  tex.SetNDC();
  tex.SetTextSize(0.07);
  tex.SetTextColor(part.color3sig);
  tex.SetTextFont(42);
  tex.DrawLatex(0.18, 0.85, part.label.Data());
  
  // Add info text
  TLatex texInfo;
  texInfo.SetNDC();
  texInfo.SetTextSize(0.030);
  texInfo.SetTextColor(kGray+2);
  texInfo.DrawLatex(0.18, 0.78, Form("Loaded %d cuts", nLoaded));
  
  gPad->RedrawAxis();
  gPad->Modified();
  gPad->Update();
  
  return canvas;
}

// =====================================================
// MAIN DRAWING FUNCTION
// =====================================================
void draw_pid_cuts_comparison(
    const TString& expFile = "pp45_pid_cuts_exp_ver1.root",
    const TString& simFile = "pp45_pid_cuts_sim_ver1.root"
) {
  gROOT->SetBatch(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  
  cout << "\n============================================================" << endl;
  cout << "  PID CUTS COMPARISON: Exp vs Sim" << endl;
  cout << "  Including: p, π+, π-, e+, e-" << endl;
  cout << "============================================================" << endl;
  
  // Open files
  TFile* fExp = TFile::Open(expFile, "READ");
  TFile* fSim = TFile::Open(simFile, "READ");
  
  if (!fExp || fExp->IsZombie()) {
    cerr << "[ERROR] Cannot open experimental file: " << expFile << endl;
    return;
  }
  if (!fSim || fSim->IsZombie()) {
    cerr << "[ERROR] Cannot open simulation file: " << simFile << endl;
    return;
  }
  
  cout << "  Exp file: " << expFile << endl;
  cout << "  Sim file: " << simFile << endl;
  
  // =====================================================
  // CONFIGURATION - 5 PARTICLES
  // =====================================================
  
  std::vector<ParticleConfig> particles = {
    // Proton
    {"p",   "Proton",   "Proton",   938.27,  4500.0, kRed-7,     kRed,       kRed+2},
    // Pion+
    {"pip", "#pi^{+}",  "#pi^{+}",  139.57,  2000.0, kBlue-7,    kBlue,      kBlue-3},
    // Pion-
    {"pim", "#pi^{-}",  "#pi^{-}",  139.57,  2000.0, kGreen+3,   kGreen+2,   kGreen-3},
    // Positron (e+)
    {"ep",  "e^{+}",    "e^{+}",    0.511,   1600.0, kMagenta-7, kMagenta,   kMagenta+2},
    // Electron (e-)
    {"em",  "e^{-}",    "e^{-}",    0.511,   1600.0, kCyan-3,    kCyan+1,    kCyan+3}
  };
  
  // Sigma levels to draw
  std::vector<SigmaConfig> sigmas = {
    {"1sig",  "1#sigma",  0, 2},
    {"3sig",  "3#sigma",  1, 3},
    {"5sig",  "5#sigma",  2, 2}
  };
  
  // Fixed Y-axis range for beta
  const double betaMin = 0.0;
  const double betaMax = 1.3;
  
  // =====================================================
  // CREATE 5 SEPARATE CANVASES - One per particle
  // =====================================================
  
  std::vector<TCanvas*> canvases;
  
  cout << "\n--- Creating individual particle canvases ---" << endl;
  
  for (size_t p = 0; p < particles.size(); ++p) {
    cout << "  Processing: " << particles[p].labelLeg << endl;
    TCanvas* c = drawParticleCanvas(fExp, fSim, particles[p], sigmas, betaMin, betaMax);
    canvases.push_back(c);
  }
  
  // =====================================================
  // CREATE SUMMARY CANVAS - All particles overlaid (3σ only)
  // =====================================================
  
  cout << "\n--- Creating summary canvas ---" << endl;
  
  TCanvas* cSummary = new TCanvas("c_pid_summary", "All PID Cuts Summary (3#sigma)", 1400, 900);
  gPad->SetLeftMargin(0.10);
  gPad->SetRightMargin(0.03);
  gPad->SetTopMargin(0.06);
  gPad->SetBottomMargin(0.10);
  gPad->SetTickx(1);
  gPad->SetTicky(1);
  
  // Frame covering all particles: 0-4500 MeV/c
  TH1F* frameAll = gPad->DrawFrame(0, betaMin, 4500, betaMax);
  frameAll->GetXaxis()->SetTitle("p [MeV/c]");
  frameAll->GetYaxis()->SetTitle("#beta");
  frameAll->GetXaxis()->SetTitleSize(0.045);
  frameAll->GetYaxis()->SetTitleSize(0.045);
  frameAll->GetXaxis()->SetLabelSize(0.040);
  frameAll->GetYaxis()->SetLabelSize(0.040);
  frameAll->GetXaxis()->SetTitleOffset(1.0);
  frameAll->GetYaxis()->SetTitleOffset(0.9);
  frameAll->GetXaxis()->SetNdivisions(510);
  frameAll->GetYaxis()->SetNdivisions(510);
  
  // Draw theoretical curves for all particles
  for (size_t p = 0; p < particles.size(); ++p) {
    TF1* fTh = new TF1(Form("fThAll_%zu", p), 
                       "x/sqrt(x*x + [0]*[0])", 0, 4500);
    fTh->SetParameter(0, particles[p].mass);
    fTh->SetLineColor(kGray);
    fTh->SetLineStyle(3);
    fTh->SetLineWidth(1);
    fTh->Draw("same");
  }
  
  // Legend for 5 particles (single column)
  TLegend* legAll = new TLegend(0.70, 0.12, 0.96, 0.55);
  legAll->SetTextSize(0.026);
  legAll->SetBorderSize(1);
  legAll->SetFillStyle(1001);
  legAll->SetNColumns(1);
  legAll->SetHeader("3#sigma cuts");
  
  // Store cuts for legend ordering
  std::vector<TCutG*> expCuts(particles.size(), nullptr);
  std::vector<TCutG*> simCuts(particles.size(), nullptr);
  
  // Draw only 3σ cuts for clarity (all particles)
  for (size_t p = 0; p < particles.size(); ++p) {
    const ParticleConfig& part = particles[p];
    TString cutName = Form("%s_cut_3sig", part.prefix.Data());
    int color = part.color3sig;
    
    // Unique clone names for combined plot
    TString expCloneName = Form("%s_exp_all_%zu", part.prefix.Data(), p);
    TString simCloneName = Form("%s_sim_all_%zu", part.prefix.Data(), p);
    
    TCutG* cutExp = loadCut(fExp, cutName, expCloneName);
    if (cutExp) {
      cutExp->SetLineColor(color);
      cutExp->SetLineWidth(3);
      cutExp->SetLineStyle(1);
      cutExp->SetFillStyle(0);
      cutExp->Draw("L same");
      expCuts[p] = cutExp;
    }
    
    TCutG* cutSim = loadCut(fSim, cutName, simCloneName);
    if (cutSim) {
      cutSim->SetLineColor(color);
      cutSim->SetLineWidth(3);
      cutSim->SetLineStyle(2);
      cutSim->SetFillStyle(0);
      cutSim->Draw("L same");
      simCuts[p] = cutSim;
    }
  }
  
  // Add legend entries: Exp then Sim for each particle
  for (size_t p = 0; p < particles.size(); ++p) {
    if (expCuts[p]) legAll->AddEntry(expCuts[p], Form("%s Exp", particles[p].labelLeg.Data()), "l");
    if (simCuts[p]) legAll->AddEntry(simCuts[p], Form("%s Sim", particles[p].labelLeg.Data()), "l");
  }
  
  legAll->Draw();
  
  // Title
  TLatex texAll;
  texAll.SetNDC();
  texAll.SetTextSize(0.04);
  texAll.SetTextFont(42);
  texAll.DrawLatex(0.12, 0.92, "All particles: 3#sigma PID cuts comparison (Exp vs Sim)");
  
  // Add particle labels at their respective positions
  TLatex texLabel;
  texLabel.SetTextSize(0.035);
  texLabel.SetTextFont(42);
  
  // Proton label (high momentum)
  texLabel.SetTextColor(particles[0].color3sig);
  texLabel.DrawLatex(3500, 0.75, "p");
  
  // Pion labels (medium momentum)
  texLabel.SetTextColor(particles[1].color3sig);
  texLabel.DrawLatex(1500, 0.98, "#pi^{+}");
  texLabel.SetTextColor(particles[2].color3sig);
  texLabel.DrawLatex(1500, 0.92, "#pi^{-}");
  
  // Lepton labels (low momentum, near β=1)
  texLabel.SetTextColor(particles[3].color3sig);
  texLabel.DrawLatex(400, 1.08, "e^{+}");
  texLabel.SetTextColor(particles[4].color3sig);
  texLabel.DrawLatex(400, 1.15, "e^{-}");
  
  gPad->RedrawAxis();
  gPad->Modified();
  gPad->Update();
  
  // =====================================================
  // SAVE CANVASES
  // =====================================================
  
  cout << "\n--- Saving canvases ---" << endl;
  
  // Save individual particle canvases
  for (size_t p = 0; p < particles.size(); ++p) {
    TString filename = Form("pid_cuts_%s_comparison.png", particles[p].prefix.Data());
    canvases[p]->SaveAs(filename);
    cout << "  Saved: " << filename << endl;
  }
  
  // Save summary canvas
  cSummary->SaveAs("pid_cuts_summary.png");
  cout << "  Saved: pid_cuts_summary.png" << endl;
  
  cout << "\n============================================================" << endl;
  cout << "  SAVED FILES:" << endl;
  cout << "    - pid_cuts_p_comparison.png   (Proton)" << endl;
  cout << "    - pid_cuts_pip_comparison.png (π+)" << endl;
  cout << "    - pid_cuts_pim_comparison.png (π-)" << endl;
  cout << "    - pid_cuts_ep_comparison.png  (e+)" << endl;
  cout << "    - pid_cuts_em_comparison.png  (e-)" << endl;
  cout << "    - pid_cuts_summary.png        (All 3σ overlaid)" << endl;
  cout << "============================================================" << endl;
  
  // Cleanup
  fExp->Close();
  fSim->Close();
}
