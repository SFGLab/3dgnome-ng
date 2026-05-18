// harness/scorer.cpp
// Calls the REAL LooperSolver scoring functions by compiling directly against
// the 3dnome MC sources. Uses "#define private public" so the private member
// functions (calcScoreHeatmapActiveRegion, etc.) are callable from test code.
// Access control is compile-time only; object layout is unchanged.
//
// Build (from repo root):
//   g++ -std=c++0x -Wno-write-strings -I3dnome/MC \
//       harness/scorer.cpp \
//       3dnome/MC/LooperSolver.cpp 3dnome/MC/Chromosome.cpp \
//       3dnome/MC/HierarchicalChromosome.cpp 3dnome/MC/Heatmap.cpp \
//       3dnome/MC/InteractionArc.cpp 3dnome/MC/InteractionArcs.cpp \
//       3dnome/MC/Cluster.cpp 3dnome/MC/ChromosomesSet.cpp \
//       3dnome/MC/BedRegion.cpp 3dnome/MC/BedRegions.cpp \
//       3dnome/MC/Anchor.cpp 3dnome/MC/Settings.cpp \
//       3dnome/MC/lib/common.cpp 3dnome/MC/lib/mtxlib.cpp \
//       3dnome/MC/lib/rmsd.cpp 3dnome/MC/lib/INIReader.cpp \
//       3dnome/MC/lib/ini.c \
//       -o harness/scorer
//
// Usage (identical to previous scorer interface):
//   ./scorer distfns <base> <scale> <power> <fscale> <fpower> <fscale_i> <fpower_i> <ca> <cscale> <cshift> <cbase>
//     stdin: lines of "genomic <bp>" | "freq <f>" | "freq_inter <f>" | "count <n>"
//   ./scorer heatmap <diagonal_size> <positions_file> <expdist_file>
//   ./scorer arcs    <stretch_k> <squeeze_k> <positions_file> <arcs_file>
//   ./scorer smooth  <stretch_k> <squeeze_k> <angular_k> <w_dist> <w_angle> <positions_file> <dtn_file>
//   ./scorer metropolis <jump_scale> <jump_coef> <score_curr> <score_prev> <T>

#define private public
#include "LooperSolver.h"
#include "Settings.h"
#include "lib/common.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

// ---------------------------------------------------------------------------
// I/O helpers (same as before)

vector<array<double,3>> read_positions(const char *path) {
    ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open positions: %s\n", path); exit(1); }
    vector<array<double,3>> pos;
    double x, y, z;
    while (f >> x >> y >> z) pos.push_back({x, y, z});
    return pos;
}

vector<vector<double>> read_matrix(const char *path) {
    ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open matrix: %s\n", path); exit(1); }
    vector<vector<double>> mat;
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        vector<double> row;
        double v;
        while (ss >> v) row.push_back(v);
        mat.push_back(row);
    }
    return mat;
}

vector<tuple<int,int,double>> read_arcs(const char *path) {
    ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open arcs: %s\n", path); exit(1); }
    vector<tuple<int,int,double>> arcs;
    int i, j; double d;
    while (f >> i >> j >> d) arcs.emplace_back(i, j, d);
    return arcs;
}

vector<double> read_dtn(const char *path) {
    ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open dtn: %s\n", path); exit(1); }
    vector<double> v;
    double x;
    while (f >> x) v.push_back(x);
    return v;
}

// ---------------------------------------------------------------------------
// Settings bootstrap
// Set only the statics needed; all others stay zero-initialized.
// Settings::dataSegmentsSplit must be non-empty to pass the constructor guard.

static void init_settings_defaults() {
    Settings::dataSegmentsSplit = "/dev/null";  // non-empty -> passes constructor check; empty file -> 0 predefined segments
    Settings::dataCentromeres   = "";
    Settings::outputLevel       = 0;

    // distance conversions - defaults matching data/GM12878/config.ini
    Settings::genomicLengthToDistBase  = 1.0f;
    Settings::genomicLengthToDistScale = 0.5f;
    Settings::genomicLengthToDistPower = 0.75f;
    Settings::freqToDistHeatmapScale      = 25.0f;
    Settings::freqToDistHeatmapPower      = -0.6f;
    Settings::freqToDistHeatmapScaleInter = 120.0f;
    Settings::freqToDistHeatmapPowerInter = -1.0f;
    Settings::countToDistA         = 0.2f;
    Settings::countToDistScale     = 1.8f;
    Settings::countToDistShift     = 8;
    Settings::countToDistBaseLevel = 0.2f;

    // spring constants - defaults from config.ini
    Settings::springConstantStretchArcs = 1.0f;
    Settings::springConstantSqueezeArcs = 1.0f;
    Settings::springConstantStretch     = 0.1f;
    Settings::springConstantSqueeze     = 0.1f;
    Settings::springAngularConstant     = 0.1f;

    // smoothness weights
    Settings::weightDistSmooth  = 1.0f;
    Settings::weightAngleSmooth = 1.0f;
}

// ---------------------------------------------------------------------------
// LooperSolver state setup helpers

static LooperSolver* make_solver() {
    return new LooperSolver("test", "/tmp/");
}

// Populate clusters + active_region from position list.
// Cluster fields not needed for scoring are left at zero/default.
static void setup_clusters(LooperSolver* ls, const vector<array<double,3>>& positions,
                            const vector<double>& dist_to_next = {}) {
    ls->clusters.clear();
    ls->active_region.clear();
    for (int i = 0; i < (int)positions.size(); i++) {
        Cluster c;
        c.pos.x = (float)positions[i][0];
        c.pos.y = (float)positions[i][1];
        c.pos.z = (float)positions[i][2];
        c.dist_to_next = (i < (int)dist_to_next.size()) ? dist_to_next[i] : 0.0;
        c.is_fixed = false;
        c.orientation = 'N';
        ls->clusters.push_back(c);
        ls->active_region.push_back(i);
    }
}

// Populate heatmap_dist (used by calcScoreHeatmapActiveRegion).
static void setup_heatmap_dist(LooperSolver* ls, const vector<vector<double>>& exp_dist, int diag_size) {
    int n = (int)exp_dist.size();
    ls->heatmap_dist.init(n);
    ls->heatmap_dist.diagonal_size = diag_size;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ls->heatmap_dist.v[i][j] = (float)exp_dist[i][j];
}

// Populate heatmap_exp_dist_anchor (used by calcScoreDistancesActiveRegion).
// Matrix entry convention: 0 = not connected (skip), < 0 = repulsion (1/d), > 0 = spring.
static void setup_arc_heatmap(LooperSolver* ls, int n,
                               const vector<tuple<int,int,double>>& arcs) {
    ls->heatmap_exp_dist_anchor.init(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ls->heatmap_exp_dist_anchor.v[i][j] = 0.0f;
    for (auto& [i, j, d] : arcs) {
        ls->heatmap_exp_dist_anchor.v[i][j] = (float)d;
        ls->heatmap_exp_dist_anchor.v[j][i] = (float)d;
    }
}

// ---------------------------------------------------------------------------
// Mode handlers - each calls the REAL LooperSolver method.

void mode_distfns(int argc, char** argv) {
    if (argc < 13) { fprintf(stderr, "distfns: needs 11 params\n"); exit(1); }
    // Override Settings with params from command line
    Settings::genomicLengthToDistBase  = atof(argv[2]);
    Settings::genomicLengthToDistScale = atof(argv[3]);
    Settings::genomicLengthToDistPower = atof(argv[4]);
    Settings::freqToDistHeatmapScale      = atof(argv[5]);
    Settings::freqToDistHeatmapPower      = atof(argv[6]);
    Settings::freqToDistHeatmapScaleInter = atof(argv[7]);
    Settings::freqToDistHeatmapPowerInter = atof(argv[8]);
    Settings::countToDistA         = atof(argv[9]);
    Settings::countToDistScale     = atof(argv[10]);
    Settings::countToDistShift     = atoi(argv[11]);
    Settings::countToDistBaseLevel = atof(argv[12]);

    // LooperSolver constructor calls freqToDistance() to populate memo table,
    // so construct AFTER setting the statics above.
    LooperSolver* ls = make_solver();

    char type[32]; double val;
    while (scanf("%31s %lf", type, &val) == 2) {
        if (strcmp(type, "genomic") == 0)
            printf("genomic %.0f -> %.10f\n", val, ls->genomicLengthToDistance((int)val));
        else if (strcmp(type, "freq") == 0)
            printf("freq %f -> %.10f\n", val, ls->freqToDistanceHeatmap((float)val));
        else if (strcmp(type, "freq_inter") == 0)
            printf("freq_inter %f -> %.10f\n", val, ls->freqToDistanceHeatmapInter((float)val));
        else if (strcmp(type, "count") == 0)
            printf("count %.0f -> %.10f\n", val, ls->freqToDistance((int)val));
        else
            fprintf(stderr, "unknown type: %s\n", type);
    }
    delete ls;
}

void mode_heatmap(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "heatmap: diagonal_size positions expdist\n"); exit(1); }
    int diag = atoi(argv[2]);
    auto pos  = read_positions(argv[3]);
    auto mat  = read_matrix(argv[4]);

    LooperSolver* ls = make_solver();
    setup_clusters(ls, pos);
    setup_heatmap_dist(ls, mat, diag);

    // calcScoreHeatmapActiveRegion() with no arg returns the full double-counted sum
    double score = ls->calcScoreHeatmapActiveRegion();
    printf("%.15f\n", score);
    delete ls;
}

void mode_arcs(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "arcs: stretch_k squeeze_k positions arcs_file\n"); exit(1); }
    Settings::springConstantStretchArcs = atof(argv[2]);
    Settings::springConstantSqueezeArcs = atof(argv[3]);
    auto pos  = read_positions(argv[4]);
    auto arcs = read_arcs(argv[5]);
    int n = (int)pos.size();

    LooperSolver* ls = make_solver();
    setup_clusters(ls, pos);
    setup_arc_heatmap(ls, n, arcs);

    // calcScoreDistancesActiveRegion() (no arg) sums over all pairs i<j
    double score = ls->calcScoreDistancesActiveRegion();
    printf("%.15f\n", score);
    delete ls;
}

void mode_smooth(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "smooth: stretch_k squeeze_k angular_k w_dist w_angle positions dtn_file\n");
        exit(1);
    }
    Settings::springConstantStretch     = atof(argv[2]);
    Settings::springConstantSqueeze     = atof(argv[3]);
    Settings::springAngularConstant     = atof(argv[4]);
    Settings::weightDistSmooth          = atof(argv[5]);
    Settings::weightAngleSmooth         = atof(argv[6]);
    auto pos = read_positions(argv[7]);
    auto dtn = read_dtn(argv[8]);

    LooperSolver* ls = make_solver();
    setup_clusters(ls, pos, dtn);

    // calcScoreStructureSmooth(lengths=true, angles=true) - the full version
    double score = ls->calcScoreStructureSmooth(true, true);
    printf("%.15f\n", score);
    delete ls;
}

void mode_metropolis(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "metropolis: jump_scale jump_coef score_curr score_prev T\n");
        exit(1);
    }
    double js = atof(argv[2]), jc = atof(argv[3]);
    double sc = atof(argv[4]), sp = atof(argv[5]), T = atof(argv[6]);
    // Matches LooperSolver.cpp line 372:
    //   tp = Settings::tempJumpScaleHeatmap * exp(-Settings::tempJumpCoefHeatmap * (score_curr/score_prev) / T)
    double prob = (T > 0.0) ? js * exp(-jc * (sc / sp) / T) : 0.0;
    printf("%.15f\n", prob);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: scorer <mode> [args...]\n");
        fprintf(stderr, "Modes: distfns | heatmap | arcs | smooth | metropolis\n");
        return 1;
    }

    init_settings_defaults();

    const char *mode = argv[1];
    if      (strcmp(mode, "distfns")    == 0) mode_distfns(argc, argv);
    else if (strcmp(mode, "heatmap")    == 0) mode_heatmap(argc, argv);
    else if (strcmp(mode, "arcs")       == 0) mode_arcs(argc, argv);
    else if (strcmp(mode, "smooth")     == 0) mode_smooth(argc, argv);
    else if (strcmp(mode, "metropolis") == 0) mode_metropolis(argc, argv);
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
    return 0;
}
