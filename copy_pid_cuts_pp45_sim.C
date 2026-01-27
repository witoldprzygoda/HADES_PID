// copy_pid_cuts.C
// Consolidates TCutG PID cuts from multiple files into one output file
// with simplified naming convention

#include "TFile.h"
#include "TCutG.h"
#include "TDirectory.h"
#include <iostream>
#include <vector>
#include <string>

void copy_pid_cuts_pp45_sim() {
    
    // Define particles and their source files
    struct ParticleInfo {
        std::string name;       // particle name (pim, pip, p)
        std::string srcFile;    // source ROOT file
    };
    
    std::vector<ParticleInfo> particles = {
        {"pim", "pim_pid_cuts_pp45_sim.root"},
        {"pip", "pip_pid_cuts_pp45_sim.root"},
        {"p",   "p_pid_cuts_pp45_sim.root"}
    };
    
    // Sigma levels to copy
    std::vector<std::string> sigmaLevels = {"1sig", "25sig", "3sig", "35sig", "5sig"};
    
    // Output file
    TFile* outFile = new TFile("pp45_pid_cuts_sim_ver1.root", "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        std::cerr << "ERROR: Cannot create output file!" << std::endl;
        return;
    }
    
    int totalCopied = 0;
    
    // Loop over particles
    for (const auto& part : particles) {
        
        TFile* srcFile = TFile::Open(part.srcFile.c_str(), "READ");
        if (!srcFile || srcFile->IsZombie()) {
            std::cerr << "WARNING: Cannot open " << part.srcFile << " - skipping" << std::endl;
            continue;
        }
        
        std::cout << "Processing " << part.srcFile << "..." << std::endl;
        
        // Navigate to TCutG directory
        TDirectory* cutDir = (TDirectory*)srcFile->Get("TCutG");
        if (!cutDir) {
            std::cerr << "  WARNING: No TCutG directory found - skipping" << std::endl;
            srcFile->Close();
            continue;
        }
        
        // Loop over sigma levels
        for (const auto& sig : sigmaLevels) {
            // Source name: cut_pim_1sig_w1, etc.
            std::string srcName = "cut_" + part.name + "_" + sig + "_w1";
            
            // Target name: pim_cut_1sig, etc.
            std::string tgtName = part.name + "_cut_" + sig;
            
            TCutG* cut = (TCutG*)cutDir->Get(srcName.c_str());
            if (!cut) {
                std::cerr << "  WARNING: " << srcName << " not found" << std::endl;
                continue;
            }
            
            // Clone and rename
            outFile->cd();
            TCutG* cutCopy = (TCutG*)cut->Clone(tgtName.c_str());
            cutCopy->SetName(tgtName.c_str());
            cutCopy->SetTitle(tgtName.c_str());
            cutCopy->Write();
            
            std::cout << "  " << srcName << " -> " << tgtName << std::endl;
            totalCopied++;
        }
        
        srcFile->Close();
    }
    
    outFile->Close();
    
    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Total cuts copied: " << totalCopied << std::endl;
    std::cout << "Output file: pp45_pid_cuts_sim_ver1.root" << std::endl;
}
