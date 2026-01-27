// =====================================================
// draw_pid_cuts_comparison.C
// =====================================================
// 
// PURPOSE: Draw TCutG contours comparing experimental and simulation
//          PID cuts for protons, π+, and π-
//
// DRAWS: 1σ, 3σ, 5σ contours
//        - Experimental: solid lines
//        - Simulation: dashed lines
//
// COLOR SCHEME:
//        - Protons (p): Red family
//        - Pions+ (pip): Blue family  
//        - Pions- (pim): Green family
//
// USAGE:
//   root -l 'draw_pid_cuts_comparison.C("pp45_pid_cuts_exp_ver1.root", "pp45_pid_cuts_sim_ver1.root")'
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
// =====================================================
TCutG* loadCut(TFile* f, const TString& cutName, const TString& cloneName) {
  if (!f || f->IsZombie()) return nullptr;
  TCutG* cut = (TCutG*)f->Get(cutName);
  if (cut) {
    cut = (TCutG*)cut->Clone(cloneName);
  }
  return cut;
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
  // CONFIGURATION
  // =====================================================
  
  // Particles and their properties
  struct ParticleConfig {
    TString prefix;      // Cut name prefix (p_, pip_, pim_)
    TString label;       // Display label
    TString labelLeg;    // Label for legend (without ROOT formatting)
    double mass;         // Particle mass [MeV/c²]
    double pMax;         // Maximum momentum for display
    int color1sig;       // Color for 1σ
    int color3sig;       // Color for 3σ
    int color5sig;       // Color for 5σ
  };
  
  // Explicit colors to ensure visibility
  std::vector<ParticleConfig> particles = {
    {"p",   "Proton",  "Proton", 938.27, 4500.0, kRed-7,    kRed,     kRed+2},
    {"pip", "#pi^{+}", "#pi^{+}", 139.57, 2000.0, kBlue-7,   kBlue,    kBlue-3},
    {"pim", "#pi^{-}", "#pi^{-}", 139.57, 2000.0, kGreen+3,  kGreen+2, kGreen-3}
  };
  
  // Sigma levels to draw
  struct SigmaConfig {
    TString suffix;      // Cut name suffix (1sig, 3sig, 5sig)
    TString label;       // Display label
    int sigmaIdx;        // Index for color selection (0=1sig, 1=3sig, 2=5sig)
    int lineWidth;       // Line width
  };
  
  std::vector<SigmaConfig> sigmas = {
    {"1sig",  "1#sigma",  0, 2},
    {"3sig",  "3#sigma",  1, 3},
    {"5sig",  "5#sigma",  2, 2}
  };
  
  // Fixed Y-axis range for beta
  const double betaMin = 0.0;
  const double betaMax = 1.3;
  
  // =====================================================
  // CREATE CANVAS - One panel per particle
  // =====================================================
  
  TCanvas* c1 = new TCanvas("c_pid_comparison", "PID Cuts: Exp vs Sim", 1800, 600);
  c1->Divide(3, 1);
  
  // Loop over particles
  for (size_t p = 0; p < particles.size(); ++p) {
    c1->cd(p + 1);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.05);
    gPad->SetTopMargin(0.08);
    gPad->SetBottomMargin(0.14);
    gPad->SetTickx(1);
    gPad->SetTicky(1);
    
    const ParticleConfig& part = particles[p];
    
    // Create frame using DrawFrame - this reliably creates axes
    TH1F* frame = gPad->DrawFrame(0, betaMin, part.pMax, betaMax);
    
    // Set axis titles with units
    frame->GetXaxis()->SetTitle("p [MeV/c]");
    frame->GetYaxis()->SetTitle("#beta");
    frame->GetXaxis()->SetTitleSize(0.055);
    frame->GetYaxis()->SetTitleSize(0.055);
    frame->GetXaxis()->SetLabelSize(0.040);   // Reduced to prevent overlap
    frame->GetYaxis()->SetLabelSize(0.045);
    frame->GetXaxis()->SetTitleOffset(1.1);
    frame->GetYaxis()->SetTitleOffset(1.1);
    frame->GetXaxis()->SetNdivisions(505);    // Fewer divisions
    frame->GetYaxis()->SetNdivisions(510);
    
    // Draw theoretical β(p) curve
    TF1* fTheory = new TF1(Form("fTheory_%zu", p), 
                           "x/sqrt(x*x + [0]*[0])", 0, part.pMax);
    fTheory->SetParameter(0, part.mass);
    fTheory->SetLineColor(kGray+1);
    fTheory->SetLineStyle(2);
    fTheory->SetLineWidth(1);
    fTheory->Draw("same");
    
    // Create legend
    TLegend* leg = new TLegend(0.50, 0.15, 0.93, 0.50);
    leg->SetTextSize(0.038);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetHeader(Form("%s cuts", part.label.Data()));
    
    // Loop over sigma levels
    for (size_t s = 0; s < sigmas.size(); ++s) {
      const SigmaConfig& sig = sigmas[s];
      
      // Construct cut names
      TString cutName = Form("%s_cut_%s", part.prefix.Data(), sig.suffix.Data());
      
      // Get color based on sigma index
      int color;
      if (sig.sigmaIdx == 0) color = part.color1sig;
      else if (sig.sigmaIdx == 1) color = part.color3sig;
      else color = part.color5sig;
      
      // Create unique clone names
      TString expCloneName = Form("%s_exp_%zu_%zu", cutName.Data(), p, s);
      TString simCloneName = Form("%s_sim_%zu_%zu", cutName.Data(), p, s);
      
      // Load and draw experimental cut
      TCutG* cutExp = loadCut(fExp, cutName, expCloneName);
      if (cutExp) {
        cutExp->SetLineColor(color);
        cutExp->SetLineWidth(sig.lineWidth);
        cutExp->SetLineStyle(1);  // Solid for exp
        cutExp->SetFillStyle(0);
        cutExp->Draw("L same");
        leg->AddEntry(cutExp, Form("Exp %s", sig.label.Data()), "l");
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
    tex.SetTextSize(0.06);
    tex.SetTextColor(part.color3sig);  // Use 3σ color as main particle color
    tex.SetTextFont(42);
    tex.DrawLatex(0.18, 0.85, part.label.Data());
    
    gPad->RedrawAxis();
    gPad->Modified();
    gPad->Update();
  }
  
  c1->Modified();
  c1->Update();
  
  // =====================================================
  // CREATE SECOND CANVAS - All particles overlaid
  // =====================================================
  
  TCanvas* c2 = new TCanvas("c_pid_all", "All PID Cuts Combined", 1200, 800);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.06);
  gPad->SetBottomMargin(0.12);
  gPad->SetTickx(1);
  gPad->SetTicky(1);
  
  // Frame covering all particles: 0-4500 MeV/c
  TH1F* frameAll = gPad->DrawFrame(0, betaMin, 4500, betaMax);
  frameAll->GetXaxis()->SetTitle("p [MeV/c]");
  frameAll->GetYaxis()->SetTitle("#beta");
  frameAll->GetXaxis()->SetTitleSize(0.05);
  frameAll->GetYaxis()->SetTitleSize(0.05);
  frameAll->GetXaxis()->SetLabelSize(0.045);
  frameAll->GetYaxis()->SetLabelSize(0.045);
  frameAll->GetXaxis()->SetTitleOffset(1.0);
  frameAll->GetYaxis()->SetTitleOffset(1.0);
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
  
  // Legend - right bottom, vertical layout (1 column)
  // Format: Proton Exp, Proton Sim, π+ Exp, π+ Sim, π- Exp, π- Sim
  TLegend* legAll = new TLegend(0.72, 0.12, 0.93, 0.42);
  legAll->SetTextSize(0.032);
  legAll->SetBorderSize(1);
  legAll->SetFillStyle(1001);
  legAll->SetNColumns(1);  // Single column for vertical layout
  legAll->SetHeader("3#sigma cuts");
  
  // Store cuts for legend ordering: Exp then Sim for each particle
  std::vector<TCutG*> expCuts(3, nullptr);
  std::vector<TCutG*> simCuts(3, nullptr);
  
  // Draw only 3σ cuts for clarity (all particles)
  for (size_t p = 0; p < particles.size(); ++p) {
    const ParticleConfig& part = particles[p];
    TString cutName = Form("%s_cut_3sig", part.prefix.Data());
    int color = part.color3sig;
    
    // Unique clone names for combined plot
    TString expCloneName = Form("%s_exp_all_%zu", cutName.Data(), p);
    TString simCloneName = Form("%s_sim_all_%zu", cutName.Data(), p);
    
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
  
  // Add legend entries in order: Proton Exp, Proton Sim, π+ Exp, π+ Sim, π- Exp, π- Sim
  for (size_t p = 0; p < particles.size(); ++p) {
    if (expCuts[p]) legAll->AddEntry(expCuts[p], Form("%s Exp", particles[p].labelLeg.Data()), "l");
    if (simCuts[p]) legAll->AddEntry(simCuts[p], Form("%s Sim", particles[p].labelLeg.Data()), "l");
  }
  
  legAll->Draw();
  
  TLatex texAll;
  texAll.SetNDC();
  texAll.SetTextSize(0.04);
  texAll.SetTextFont(42);
  texAll.DrawLatex(0.15, 0.88, "All particles: 3#sigma cuts");
  
  gPad->RedrawAxis();
  gPad->Modified();
  gPad->Update();
  
  // =====================================================
  // SAVE CANVASES
  // =====================================================
  
  //c1->SaveAs("pid_cuts_comparison.png");
  //c2->SaveAs("pid_cuts_all_particles.png");
  
  //cout << "\n============================================================" << endl;
  //cout << "  SAVED FILES:" << endl;
  //cout << "    - pid_cuts_comparison.png (separate panels)" << endl;
  //cout << "    - pid_cuts_all_particles.png (all 3σ overlaid)" << endl;
  //cout << "============================================================" << endl;
  
  // Cleanup
  fExp->Close();
  fSim->Close();
}
