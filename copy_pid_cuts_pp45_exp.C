// ROOT macro to copy and rename TCutG objects from PID systematics files
// ADAPTED FOR PION BEAM DATA
// Usage: root -l -b -q copy_pid_cuts_pp45.C
void copy_pid_cuts_pp45_exp() {
    // Define input files and their corresponding particle prefixes
    std::vector<std::pair<TString, TString>> files = {
        {"pid_p_systematics.root",   "p"},
        {"pid_pip_systematics.root", "pip"},
        {"pid_pim_systematics.root", "pim"}
    };
    
    // Sigma levels to process - all 5 levels stored in files
    std::vector<TString> sigmas = {"1sig", "25sig", "3sig", "35sig", "5sig"};
    
    // Create output file
    TFile *outFile = new TFile("pp45_pid_cuts_ver1.root", "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        std::cerr << "Error: Cannot create output file!" << std::endl;
        return;
    }
    
    // Process each input file
    for (const auto &filePair : files) {
        TString inputFileName = filePair.first;
        TString particle = filePair.second;
        
        // Open input file
        TFile *inFile = TFile::Open(inputFileName, "READ");
        if (!inFile || inFile->IsZombie()) {
            std::cerr << "Warning: Cannot open " << inputFileName << ", skipping..." << std::endl;
            continue;
        }
        
        std::cout << "Processing: " << inputFileName << std::endl;
        
        // Navigate to EnvelopeCuts directory
        TDirectoryFile *dir = (TDirectoryFile*)inFile->Get("EnvelopeCuts");
        if (!dir) {
            std::cerr << "Warning: EnvelopeCuts directory not found in " << inputFileName << std::endl;
            inFile->Close();
            continue;
        }
        
        // Process each sigma level
        for (const auto &sig : sigmas) {
            // Construct original object name (e.g., "envelope_p_3sig")
            TString origName = Form("envelope_%s_%s", particle.Data(), sig.Data());
            
            // Get the TCutG object
            TCutG *cut = (TCutG*)dir->Get(origName);
            if (!cut) {
                std::cerr << "  Warning: " << origName << " not found, skipping..." << std::endl;
                continue;
            }
            
            // Create new name (e.g., "p_cut_3sig")
            TString newName = Form("%s_cut_%s", particle.Data(), sig.Data());
            
            // Clone and rename
            outFile->cd();
            TCutG *cutCopy = (TCutG*)cut->Clone(newName);
            cutCopy->SetTitle(newName);
            cutCopy->Write();
            
            std::cout << "  Copied: " << origName << " -> " << newName << std::endl;
        }
        
        inFile->Close();
    }
    
    // Save and close output file
    outFile->Close();
    
    std::cout << "\nOutput file created: pp45_pid_cuts_ver1.root" << std::endl;
    std::cout << "Contents: p, pip, pim cuts at 1σ, 2.5σ, 3σ, 3.5σ, 5σ levels" << std::endl;
    std::cout << "Done!" << std::endl;
}
