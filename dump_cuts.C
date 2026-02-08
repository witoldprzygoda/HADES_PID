// dump_cuts.C
// Run on the machine where the file opens correctly (ROOT 6.32):
//   root -l -b -q 'dump_cuts.C("pp45_pid_cuts_exp_ver2.root", "cuts_dump.txt")'

void dump_cuts(const char* inFile = "pp45_pid_cuts_exp_ver2.root",
               const char* outTxt = "cuts_dump.txt")
{
    TFile *fin = TFile::Open(inFile);
    if (!fin || fin->IsZombie()) {
        printf("Cannot open %s\n", inFile);
        return;
    }

    FILE *fp = fopen(outTxt, "w");

    TIter next(fin->GetListOfKeys());
    TKey *key;
    while ((key = (TKey*)next())) {
        if (TString(key->GetClassName()) != "TCutG") continue;

        TCutG *cut = (TCutG*)key->ReadObj();
        int np = cut->GetN();

        // Header: name  title  varx  vary  npoints
        fprintf(fp, "# %s | %s | %s | %s | %d\n",
                cut->GetName(),
                cut->GetTitle(),
                cut->GetVarX(),
                cut->GetVarY(),
                np);

        // Data: x  y
        for (int j = 0; j < np; j++) {
            Double_t x, y;
            cut->GetPoint(j, x, y);
            fprintf(fp, "%.15g %.15g\n", x, y);
        }

        fprintf(fp, "\n");
        printf("Dumped: %s (%d points)\n", cut->GetName(), np);
    }

    fclose(fp);
    fin->Close();
    printf("\nAll cuts written to %s\n", outTxt);
}
