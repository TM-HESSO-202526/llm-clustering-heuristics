//////////////////////////////////////////////////////////////
// Clustering avec pour objectif la minimisation de le volume de 
// p hypersphères couvrant tous les éléments.
// Deux méthodes semblables à k-means et PAM implémentées + hybride échantillonné
// Autheur: © E. Taillard
// Date: 2026/05/14
// Compilation: g++ -O2 clustering_sphere.cpp -o clustering_sphere.exe
// Utilisation: clustering_sphere.exe data_filename option number_of_runs\n"
// Pour k-means, donner l'option 0;  PAM (beaucoup plus lent) 1
////////////////////////////////////////////////////////////////

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>

using namespace std;

random_device rd;
mt19937 gen(rd());

/////////////////// Objectif spécifique: minimiser le volume des hypersphères

// Distance euclidienne au carré entre a et b
double sed(const vector<int>& a, const vector<int>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Qualité du placement du centre sur center pour un groupe.
// Retourne le carré du rayon du groupe
double cluster_cost(const vector<vector<int>>& data, 
                    const vector<size_t> & cluster, size_t center) {
   double res = 0.0;
   for (auto e : cluster) {
     double d = sed(data[e], data[center]);
     if (res < d)
       res = d;
   }
   return res;
}

// Coût de la solution (proportionnel au volume des sphères)
double total_cost(const vector<vector<int>>& data, 
                  const vector<size_t>& assignment) {
  size_t n = data.size();
  vector<double> plus_eloigne(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    double d = pow(sed(data[i], data[assignment[i]]), 
                   double(data[i].size()) / 2.0);
    if (plus_eloigne[assignment[i]] < d)
      plus_eloigne[assignment[i]] = d;
  }
  return accumulate(plus_eloigne.begin(), plus_eloigne.end(), 0.0);
}

//////////////////// Fin de la partie spécifique

// Placement aléatoire des centres
vector<size_t> gen_sol_init(size_t n, size_t p) {
  vector<size_t> res(n);
  iota(res.begin(), res.end(), 0);
  shuffle(res.begin(), res.end(), gen);
  res.resize(p);
  return res;
}

// Affectation de chaque élément vers son plus proche centre
void assign(const vector<vector<int>>& data, 
            const vector<size_t> & medoids, 
            vector<size_t> & assignment) {
  size_t n = data.size(), p = medoids.size();
  vector<double> dist_min(n, numeric_limits<double>::max());
  for (size_t i = 0; i < n; ++i)  
    for (size_t j = 0; j < p; ++j) {
      double d = sed(data[i], data[medoids[j]]);
      if (d < dist_min[i]) {
        dist_min[i] = d;
        assignment[i] = medoids[j];
      }
    }
}

/////////////////////// Méthode type k-means /////////////////////
// Repositionner chaque centre au mieux parmi les éléments lui étant rattaché.
// Retourne vrai si modification
bool reposition(const vector<vector<int>>& data, 
                vector<size_t> & medoids, 
                const vector<size_t> & assignment) {
  bool res = false;
  size_t n = data.size(), p = medoids.size();
  vector<size_t> cluster; cluster.reserve(n);
  for (size_t j = 0; j < p; ++j) { // reposition medoid for cluster j
    // Find jth cluster
    cluster.clear();
    for (size_t i = 0; i < n; ++i) if (assignment[i] == medoids[j]) 
      cluster.push_back(i);
    double old_cost = cluster_cost(data, cluster, medoids[j]), 
           new_cost = (1.0 - 1e-10) * old_cost; // to avoid numerical issues

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

// Méthode type k-means. Retourne le coût global
// Modifie les médoïdes et l'affectation des éléments au médoïde le plus proche
double kmedian(const vector<vector<int>>& data, 
               vector<size_t> & medoids,
               vector<size_t> & assignment) {
    size_t n = data.size(), p = medoids.size();
    if (p <= 0 || p > n) return {};

    do 
        assign(data, medoids, assignment);
    while (reposition(data, medoids, assignment) );

    return total_cost(data, assignment);
}

//////////////////// Méthode type Partition Around Medoids
// Méthode type pam. Retourne le coût global
// Modifie les médoïdes et l'affectation des éléments au médoïde le plus proche
double pam(const vector<vector<int>>& data, 
           vector<size_t> & medoids,
           vector<size_t> & assignment) {
    size_t n = data.size(), p = medoids.size();
    if (p <= 0 || p > n) return {};
    assign(data, medoids, assignment);
    double best_cost = total_cost(data, assignment);
    vector<bool> is_medoid(n);
    for (const auto & m : medoids)
      is_medoid[m] = true;
    for (size_t j = 0; j < p; ++j) {
      for (size_t i = 0; i < n; ++i) if (not is_medoid[i]) {
        // Déplacer le médoïde j sur l'élément i
        auto old_medoid = medoids[j];
        medoids[j] = i;
        assign(data, medoids, assignment);
        double new_cost = total_cost(data, assignment);
        if (new_cost < (1.0 - 1e-10) * best_cost) {
          is_medoid[old_medoid] = false;
          is_medoid[i] = true;
          best_cost = new_cost;
        }
        else
          medoids[j] = old_medoid; // pas amélioré, on remet j en place
      }
    }
  assign(data, medoids, assignment);
  return best_cost;
}

///////////////// Méthode hybride: PAM sur échantillon initialisant kmedian
// Au maximum 10 éléments par centres retenus dans l'échantillon
double hybrid_pam_kmedian(const vector<vector<int>>& data, 
           vector<size_t> & medoids,
           vector<size_t> & assignment) {
    size_t n = data.size(), p = medoids.size(), ns = 10*p;
    if (p <= 0 || p > n) return {};
    if (n <= ns) 
      return pam(data, medoids, assignment);
      
    auto id_sample = gen_sol_init(n, ns);
    vector<vector<int>> sample(data.begin(), data.begin() + ptrdiff_t(ns));
    for (size_t i = 0; i < ns; ++i)
      sample[i] = data[id_sample[i]];
    auto medoids_s = gen_sol_init(ns, p);
    pam(sample, medoids_s, assignment);
    for (size_t j = 0; j < p; ++j)
      medoids[j] = id_sample[medoids_s[j]];
    return kmedian(data, medoids, assignment);
}

int main(int argc, char*argv[]) {
    if (argc != 4) {
      cerr << "Usage: " << argv[0] << " data_filename option number_of_runs\n"
      << "Option 0: k-means, 1: PAM, 2: PAM on sample + k-means\n";
      exit(-1);
    }
    ifstream in(argv[1]);
        
    size_t n, dim, p; // nombre d'éléments, dimension, nombre de centres
    double bidon_double;
    in >> n >> p >> dim >> bidon_double >> bidon_double;
    // Coordonnées des éléments
    vector<vector<int>> data(n, vector<int> (dim)); 
    // Lecture du fichier
    for (auto & e : data)
      for (auto & c : e)
        in >> c;
    
    vector<size_t> medoids(p);
    vector<size_t> assignment(n);
    chrono::high_resolution_clock::time_point t;
    for (size_t j = 0; j < p; ++j)
      medoids[j] = n + j - p;
    
    cout << "Reference (value, time[s]): ";
    t = chrono::high_resolution_clock::now();
    assign(data, medoids, assignment);
    double reference = total_cost(data, assignment);
    cout << reference << ' ' 
         << chrono::duration<double>(chrono::high_resolution_clock::now() - t).count()
         << endl;
    cout << "Reference improved with PAM (value/reference, time[s]): ";
    t = chrono::high_resolution_clock::now();
    double best_sol = pam(data, medoids, assignment);
    cout  << best_sol / reference << ' ' 
          << chrono::duration<double>(chrono::high_resolution_clock::now() - t).count()
          << endl;

  // Répétition de résolutions avec affichage à chaque amélioration
    for (size_t no_sol = 0; no_sol < stoul(argv[3]); ++no_sol) {
        medoids = gen_sol_init(n, p);
        double current_cost;
        int method = stoi(argv[2]);
        t = chrono::high_resolution_clock::now();
        if (method == 0)
          current_cost = kmedian(data, medoids, assignment);
        else if (method == 1)
          current_cost = pam(data, medoids, assignment);
        else
          current_cost = hybrid_pam_kmedian(data, medoids, assignment);
        cout << (current_cost )/ reference << ' '
             << chrono::duration<double>(chrono::high_resolution_clock::now() - t).count()
             << endl;
        if (best_sol > current_cost) {
          best_sol = current_cost;
          cout << no_sol << " " << current_cost << " Best sol improved; medoids :\n";
          for (auto idx : medoids) {
              cout << idx << ' ';
              for (auto c : data[idx]) cout << c << " ";
              cout << endl;
          }
        }
    }
}
