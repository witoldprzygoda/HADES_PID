// restore_cuts.C
// Run on the target machine (ROOT 6.24):
//   root -l -b -q 'restore_cuts.C("cuts_dump.txt", "pp45_pid_cuts_exp_ver2_new.root")'

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void restore_cuts(const char* inTxt  = "cuts_dump.txt",
                  const char* outFile = "pp45_pid_cuts_sim_ver3.root")
{
    std::ifstream fin(inTxt);
    if (!fin.is_open()) {
        printf("Cannot open %s\n", inTxt);
        return;
    }

    TFile *fout = new TFile(outFile, "RECREATE");
    int nCuts = 0;

    std::string line;
    while (std::getline(fin, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Parse header line: # name | title | varx | vary | npoints
        if (line[0] != '#') continue;

        // Remove leading "# "
        std::string header = line.substr(2);

        // Split by " | "
        std::vector<std::string> fields;
        size_t pos = 0;
        std::string delim = " | ";
        while ((pos = header.find(delim)) != std::string::npos) {
            fields.push_back(header.substr(0, pos));
            header.erase(0, pos + delim.length());
        }
        fields.push_back(header);

        if (fields.size() != 5) {
            printf("Bad header: %s\n", line.c_str());
            continue;
        }

        std::string name  = fields[0];
        std::string title = fields[1];
        std::string varx  = fields[2];
        std::string vary  = fields[3];
        int np = std::stoi(fields[4]);

        // Read points
        std::vector<Double_t> xv(np), yv(np);
        for (int j = 0; j < np; j++) {
            if (!std::getline(fin, line)) {
                printf("Unexpected EOF reading %s point %d\n", name.c_str(), j);
                fout->Close();
                return;
            }
            std::istringstream iss(line);
            iss >> xv[j] >> yv[j];
        }

        // Create TCutG
        fout->cd();
        TCutG *cut = new TCutG(name.c_str(), np, xv.data(), yv.data());
        cut->SetTitle(title.c_str());
        cut->SetVarX(varx.c_str());
        cut->SetVarY(vary.c_str());
        cut->Write();

        // Remove from gROOT specials to avoid cleanup issues
        gROOT->GetListOfSpecials()->Remove(cut);

        printf("Restored: %s (%d points)\n", name.c_str(), np);
        nCuts++;
    }

    fin.close();
    fout->Close();
    printf("\nDone. Written %d cuts to %s\n", nCuts, outFile);
}
