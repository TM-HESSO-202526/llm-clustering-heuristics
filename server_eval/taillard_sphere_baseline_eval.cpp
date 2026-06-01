//////////////////////////////////////////////////////////////
// Taillard radius/coverage baseline evaluator for server_eval.
// Derived from external/taillard_cpp/clustering_sphere.cpp.
// It exposes exactly the three options used for the final external baselines:
//   0: k-means/k-median-like medoid refinement
//   1: PAM
//   2: PAM on sample + k-means/k-median-like refinement
//
// Difference from the original demonstrator: this executable performs only
// the requested method once and prints machine-parseable COST/TIME_S lines.
// Compilation: g++ -O2 -std=c++17 taillard_sphere_baseline_eval.cpp -o taillard_sphere_baseline_eval
// Usage: taillard_sphere_baseline_eval data_filename option seed
//////////////////////////////////////////////////////////////

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <string>

using namespace std;

static mt19937 gen;

// Squared Euclidean distance between a and b.
double sed(const vector<int>& a, const vector<int>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return sum;
}

// Quality of placing the cluster center on center. Returns squared radius.
double cluster_cost(const vector<vector<int>>& data, const vector<size_t>& cluster, size_t center) {
    double res = 0.0;
    for (auto e : cluster) {
        double d = sed(data[e], data[center]);
        if (res < d) res = d;
    }
    return res;
}

// Objective: sum over centers of radius^dimension, equivalent to original volume-proportional cost.
double total_cost(const vector<vector<int>>& data, const vector<size_t>& assignment) {
    size_t n = data.size();
    vector<double> farthest_power(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double d = pow(sed(data[i], data[assignment[i]]), double(data[i].size()) / 2.0);
        if (farthest_power[assignment[i]] < d) farthest_power[assignment[i]] = d;
    }
    return accumulate(farthest_power.begin(), farthest_power.end(), 0.0);
}

vector<size_t> gen_sol_init(size_t n, size_t p) {
    vector<size_t> res(n);
    iota(res.begin(), res.end(), 0);
    shuffle(res.begin(), res.end(), gen);
    res.resize(p);
    return res;
}

void assign(const vector<vector<int>>& data, const vector<size_t>& medoids, vector<size_t>& assignment) {
    size_t n = data.size(), p = medoids.size();
    vector<double> dist_min(n, numeric_limits<double>::max());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double d = sed(data[i], data[medoids[j]]);
            if (d < dist_min[i]) {
                dist_min[i] = d;
                assignment[i] = medoids[j];
            }
        }
    }
}

bool reposition(const vector<vector<int>>& data, vector<size_t>& medoids, const vector<size_t>& assignment) {
    bool res = false;
    size_t n = data.size(), p = medoids.size();
    vector<size_t> cluster; cluster.reserve(n);
    for (size_t j = 0; j < p; ++j) {
        cluster.clear();
        for (size_t i = 0; i < n; ++i) if (assignment[i] == medoids[j]) cluster.push_back(i);
        if (cluster.empty()) continue;
        double old_cost = cluster_cost(data, cluster, medoids[j]);
        double new_cost = (1.0 - 1e-10) * old_cost;
        for (size_t i = 0; i < cluster.size(); ++i) if (cluster[i] != medoids[j]) {
            double nc = cluster_cost(data, cluster, cluster[i]);
            if (nc < new_cost) {
                new_cost = nc;
                medoids[j] = cluster[i];
                res = true;
            }
        }
    }
    return res;
}

double kmedian(const vector<vector<int>>& data, vector<size_t>& medoids, vector<size_t>& assignment) {
    size_t n = data.size(), p = medoids.size();
    if (p <= 0 || p > n) return 0.0;
    do {
        assign(data, medoids, assignment);
    } while (reposition(data, medoids, assignment));
    return total_cost(data, assignment);
}

double pam(const vector<vector<int>>& data, vector<size_t>& medoids, vector<size_t>& assignment) {
    size_t n = data.size(), p = medoids.size();
    if (p <= 0 || p > n) return 0.0;
    assign(data, medoids, assignment);
    double best_cost = total_cost(data, assignment);
    vector<bool> is_medoid(n, false);
    for (const auto& m : medoids) is_medoid[m] = true;
    for (size_t j = 0; j < p; ++j) {
        for (size_t i = 0; i < n; ++i) if (!is_medoid[i]) {
            auto old_medoid = medoids[j];
            medoids[j] = i;
            assign(data, medoids, assignment);
            double new_cost = total_cost(data, assignment);
            if (new_cost < (1.0 - 1e-10) * best_cost) {
                is_medoid[old_medoid] = false;
                is_medoid[i] = true;
                best_cost = new_cost;
            } else {
                medoids[j] = old_medoid;
            }
        }
    }
    assign(data, medoids, assignment);
    return best_cost;
}

double hybrid_pam_kmedian(const vector<vector<int>>& data, vector<size_t>& medoids, vector<size_t>& assignment) {
    size_t n = data.size(), p = medoids.size(), ns = 10 * p;
    if (p <= 0 || p > n) return 0.0;
    if (n <= ns) return pam(data, medoids, assignment);
    auto id_sample = gen_sol_init(n, ns);
    vector<vector<int>> sample(ns);
    for (size_t i = 0; i < ns; ++i) sample[i] = data[id_sample[i]];
    vector<size_t> assignment_s(ns);
    auto medoids_s = gen_sol_init(ns, p);
    pam(sample, medoids_s, assignment_s);
    for (size_t j = 0; j < p; ++j) medoids[j] = id_sample[medoids_s[j]];
    return kmedian(data, medoids, assignment);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " data_filename option seed\n"
             << "Option 0: k-means/k-median-like, 1: PAM, 2: PAM on sample + k-means/k-median-like\n";
        return 2;
    }
    string filename = argv[1];
    int method = stoi(argv[2]);
    unsigned int seed = static_cast<unsigned int>(stoul(argv[3]));
    gen.seed(seed);

    ifstream in(filename);
    if (!in) {
        cerr << "Could not open input file: " << filename << "\n";
        return 3;
    }

    size_t n, dim, p;
    double dummy1 = 0.0, dummy2 = 0.0;
    in >> n >> p >> dim >> dummy1 >> dummy2;
    if (!in || n == 0 || p == 0 || dim == 0) {
        cerr << "Could not read header n p dim dummy dummy from " << filename << "\n";
        return 4;
    }

    vector<vector<int>> data(n, vector<int>(dim));
    for (auto& e : data) {
        for (auto& c : e) in >> c;
    }
    if (!in) {
        cerr << "Could not read all coordinates from " << filename << "\n";
        return 5;
    }

    vector<size_t> medoids = gen_sol_init(n, p);
    vector<size_t> assignment(n);

    auto t0 = chrono::high_resolution_clock::now();
    double cost = 0.0;
    if (method == 0) cost = kmedian(data, medoids, assignment);
    else if (method == 1) cost = pam(data, medoids, assignment);
    else if (method == 2) cost = hybrid_pam_kmedian(data, medoids, assignment);
    else {
        cerr << "Unknown option: " << method << "\n";
        return 6;
    }
    double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();

    cout.precision(17);
    cout << "COST " << cost << "\n";
    cout << "TIME_S " << elapsed << "\n";
    return 0;
}
