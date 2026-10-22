// PopTree (SkipPopList form) from Iacono & Pareek, arXiv 2610.29574, Section 3,
// plus a splay tree, a static balanced BST, and the paper's OPT(X) formula.
// Everything is measured as "nodes touched per search" in a binary search tree.
//
// build: g++ -O2 -std=c++17 -o poptree poptree.cpp
// usage: ./poptree <L> <m> <workload> <seed> [check]
//   n = 3^L keys (1..n), m searches. "check" verifies every invariant after every search.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <climits>
#include <array>

using namespace std;
typedef long long ll;

static int L;          // levels 1..L hold buckets; level L+1 (bottom) stores every key
static ll n;
static vector<ll> pow3;
static bool fill_first = false;   // ablation: gap repair takes the leftmost occupied child instead

// ---------------------------------------------------------------- SkipPopList
struct SkipPopList {
    vector<vector<int>> occ;   // occ[l][b] = key occupying bucket b at level l (0 = vacant), l in 1..L
    vector<int> top;           // top[x] = smallest level that stores x (L+1 = bottom only)
    vector<ll> last;           // last search time of x (-1 = never), used by the checker only
    ll demotions = 0, promotions = 0;

    ll bsize(int l) const { return pow3[L - l]; }           // keys per bucket at level l
    ll nb(int l) const { return pow3[l]; }                   // buckets at level l
    ll bucket(int l, ll x) const { return (x - 1) / bsize(l); }
    bool extreme(int l, ll b) const { return b == 0 || b == nb(l) - 1; }

    void init() {
        occ.assign(L + 2, {});
        for (int l = 1; l <= L; l++) occ[l].assign(nb(l), 0);
        top.assign(n + 1, L + 1);
        last.assign(n + 1, -1);
        // Level L: every bucket is one key; occupy buckets 1 mod 3 (vacant extremes, gaps of 2).
        // Level l < L: bucket b takes the occupant of its middle child 3b+1, so the same pattern repeats.
        for (ll b = 1; b < nb(L); b += 3) { occ[L][b] = (int)(b + 1); top[b + 1] = L; }
        for (int l = L - 1; l >= 1; l--)
            for (ll b = 1; b < nb(l); b += 3) {
                int y = occ[l + 1][3 * b + 1];
                occ[l][b] = y; top[y] = l;
            }
    }

    void demote_top(int y) {          // remove y from its topmost level
        int l = top[y];
        occ[l][bucket(l, y)] = 0;
        top[y] = l + 1;
        demotions++;
    }

    void search(ll x, ll time) {
        int li = top[(int)x];
        int t = 1;
        while (t <= L && extreme(t, bucket(t, x))) t++;     // highest non-extreme level for x
        if (t < li) {
            // clearing demotions, levels [t, li)
            for (int l = t; l < li; l++) {
                ll B = bucket(l, x);
                for (ll d = -1; d <= 1; d++) {
                    ll b = B + d;
                    if (b < 0 || b >= nb(l)) continue;
                    int y = occ[l][b];
                    if (y == 0 || y == x) continue;
                    // demote y until no level it sits on has its bucket within 1 of x's bucket
                    while (top[y] <= L && llabs(bucket(top[y], y) - bucket(top[y], x)) <= 1) demote_top(y);
                }
            }
            // promotion of x
            for (int l = li - 1; l >= t; l--) { occ[l][bucket(l, x)] = (int)x; promotions++; }
            top[(int)x] = t;
            // gap repair, bottom-up; any run of >= 3 vacant buckets lives within a few buckets of x
            for (int l = L; l >= 1; l--) {
                ll B = bucket(l, x);
                ll lo = max(0LL, B - 6), hi = min(nb(l) - 1, B + 6);
                ll b = lo;
                while (b + 2 <= hi) {
                    if (occ[l][b] == 0 && occ[l][b + 1] == 0 && occ[l][b + 2] == 0) {
                        fill(l, b + 1);
                        b += 2;
                    } else b++;
                }
            }
        }
        last[(int)x] = time;
    }

    // promote into vacant bucket b at level l the occupant of one of its children
    // (the most recently searched one; the paper allows any).
    void fill(int l, ll b) {
        int best = 0;
        for (ll c = 3 * b; c < 3 * b + 3; c++) {
            int y;
            if (l + 1 <= L) y = occ[l + 1][c];
            else y = (c % 3 == 1) ? (int)(c / 3 + 1) : 0;   // bottom level: key in middle child
            if (y && (best == 0 || (!fill_first && last[y] > last[best]))) best = y;
        }
        if (!best) { fprintf(stderr, "gap repair found no child occupant (l=%d b=%lld)\n", l, b); exit(2); }
        if (top[best] != l + 1) { fprintf(stderr, "child occupant not topmost below\n"); exit(2); }
        occ[l][b] = best; top[best] = l; promotions++;
    }

    // Nodes touched to find x in the BST obtained from this skip list: the level-1 key is the
    // root; between two consecutive keys of level l, the keys new at level l+1 form a group
    // laid out as a median-split balanced BST (the usual B-tree -> BST conversion).
    int bst_cost(ll x, vector<int>* path = nullptr) const {
        ll lo = 0, hi = n + 1;
        int cost = 0;
        static vector<int> g;
        for (int l = 1; l <= L + 1; l++) {
            g.clear();
            if (l <= L) {
                ll b0 = (lo < 1) ? 0 : bucket(l, lo), b1 = (hi > n) ? nb(l) - 1 : bucket(l, hi);
                for (ll b = b0; b <= b1; b++) { int y = occ[l][b]; if (y > lo && y < hi) g.push_back(y); }
            } else {
                for (ll y = lo + 1; y < hi; y++) g.push_back((int)y);
            }
            // descend the balanced BST over g
            int a = 0, z = (int)g.size() - 1;
            ll nlo = lo, nhi = hi;
            while (a <= z) {
                int mid = (a + z + 1) / 2;
                cost++;
                if (path) path->push_back(g[mid]);
                if (g[mid] == x) return cost;
                if (x < g[mid]) { nhi = g[mid]; z = mid - 1; } else { nlo = g[mid]; a = mid + 1; }
            }
            lo = nlo; hi = nhi;
        }
        fprintf(stderr, "bst_cost: key %lld not found\n", x); exit(2);
    }

    void check(ll time) const {
        for (int l = 1; l <= L; l++) {
            const vector<int>& o = occ[l];
            if (o[0] || o[nb(l) - 1]) { fprintf(stderr, "extreme occupied l=%d t=%lld\n", l, time); exit(3); }
            int run = 0;
            for (ll b = 0; b < nb(l); b++) {
                if (o[b]) {
                    if (b && o[b - 1]) { fprintf(stderr, "adjacent occupied l=%d\n", l); exit(3); }
                    if (bucket(l, o[b]) != b) { fprintf(stderr, "key in wrong bucket\n"); exit(3); }
                    if (top[o[b]] > l) { fprintf(stderr, "top wrong\n"); exit(3); }
                    run = 0;
                } else if (++run >= 3) { fprintf(stderr, "3 vacant l=%d b=%lld t=%lld\n", l, b, time); exit(3); }
            }
        }
        for (ll x = 1; x <= n; x++)
            for (int l = top[x]; l <= L; l++)
                if (occ[l][bucket(l, x)] != x) { fprintf(stderr, "subset property broken\n"); exit(3); }
        // temporal: an important bucket holds its most recently searched key
        for (int l = 1; l <= L; l++) {
            vector<ll> mrt(nb(l), -1); vector<int> mrk(nb(l), 0);
            for (ll x = 1; x <= n; x++) {
                ll b = bucket(l, x);
                if (last[x] > mrt[b]) { mrt[b] = last[x]; mrk[b] = (int)x; }
            }
            for (ll b = 1; b + 1 < nb(l); b++) {
                if (mrt[b] < 0) continue;
                // siblings = the two adjacent buckets (as used in the proofs of Lemmas 20 and 23)
                bool imp = mrt[b - 1] < mrt[b] && mrt[b + 1] < mrt[b];
                if (imp && occ[l][b] != mrk[b]) { fprintf(stderr, "important bucket not held by its MRU key l=%d t=%lld\n", l, time); exit(3); }
            }
        }
    }

    // smallest alpha over the converted BST: min over nodes of w(child)/w(parent), w = size+1
    double min_alpha() const {
        // build explicit BST by inserting along bst_cost paths is O(n^2); instead recurse on groups
        double best = 1.0;
        rec(1, 0, n + 1, best);
        return best;
    }
    // returns weight (size+1) of the subtree spanning keys in (lo,hi) starting at level l
    ll rec(int l, ll lo, ll hi, double& best) const {
        if (hi - lo - 1 <= 0) return 1;
        vector<int> g;
        if (l <= L) {
            ll b0 = (lo < 1) ? 0 : bucket(l, lo), b1 = (hi > n) ? nb(l) - 1 : bucket(l, hi);
            for (ll b = b0; b <= b1; b++) { int y = occ[l][b]; if (y > lo && y < hi) g.push_back(y); }
        } else for (ll y = lo + 1; y < hi; y++) g.push_back((int)y);
        if (g.empty()) return rec(l + 1, lo, hi, best);
        return rec_group(l, g, 0, (int)g.size() - 1, lo, hi, best);
    }
    ll rec_group(int l, const vector<int>& g, int a, int z, ll lo, ll hi, double& best) const {
        if (a > z) return rec(l + 1, lo, hi, best);
        int mid = (a + z + 1) / 2;
        ll wl = rec_group(l, g, a, mid - 1, lo, g[mid], best);
        ll wr = rec_group(l, g, mid + 1, z, g[mid], hi, best);
        ll w = wl + wr;
        best = min(best, (double)min(wl, wr) / (double)w);
        return w;
    }
};

// ---------------------------------------------------------------- splay tree
struct Splay {
    vector<int> L_, R_, P_;
    int root;
    int build(int lo, int hi, int par) {
        if (lo > hi) return 0;
        int mid = (lo + hi) / 2;
        P_[mid] = par;
        L_[mid] = build(lo, mid - 1, mid);
        R_[mid] = build(mid + 1, hi, mid);
        return mid;
    }
    void init() { L_.assign(n + 1, 0); R_.assign(n + 1, 0); P_.assign(n + 1, 0); root = build(1, (int)n, 0); }
    void rot(int x) {
        int p = P_[x], g = P_[p];
        if (L_[p] == x) { L_[p] = R_[x]; if (R_[x]) P_[R_[x]] = p; R_[x] = p; }
        else { R_[p] = L_[x]; if (L_[x]) P_[L_[x]] = p; L_[x] = p; }
        P_[p] = x; P_[x] = g;
        if (g) { if (L_[g] == p) L_[g] = x; else R_[g] = x; } else root = x;
    }
    int access(int x) {             // returns nodes touched (depth + 1), then splays x to root
        int d = 1;
        for (int v = root; v != x; v = (x < v) ? L_[v] : R_[v]) d++;
        while (P_[x]) {
            int p = P_[x], g = P_[p];
            if (g) rot(((L_[g] == p) == (L_[p] == x)) ? p : x);
            rot(x);
        }
        return d;
    }
};

// static perfectly balanced BST on 1..n (same shape as Splay::build)
static int static_cost(ll x) {
    ll lo = 1, hi = n; int d = 0;
    while (true) { ll mid = (lo + hi) / 2; d++; if (mid == x) return d; if (x < mid) hi = mid - 1; else lo = mid + 1; }
}

// static BST built offline from the whole sequence's key frequencies, with no balance rule:
// the root splits the total weight in half, recursively (Mehlhorn's bisection rule, within
// a small additive constant of the entropy bound). Weight = count + 0.01 so unseen keys sit deep.
static vector<int> entropy_depth(const vector<int>& X) {
    vector<double> pre(n + 1, 0.0);
    vector<ll> cnt(n + 1, 0);
    for (int x : X) cnt[x]++;
    for (ll x = 1; x <= n; x++) pre[x] = pre[x - 1] + cnt[x] + 0.01;
    vector<int> dep(n + 1, 0);
    // iterative recursion over (lo, hi, depth)
    vector<array<ll, 3>> stk; stk.push_back({1, n, 1});
    while (!stk.empty()) {
        auto [lo, hi, d] = stk.back(); stk.pop_back();
        if (lo > hi) continue;
        double mid = (pre[lo - 1] + pre[hi]) / 2;
        // smallest k in [lo,hi] with pre[k] >= mid
        ll k = lower_bound(pre.begin() + lo, pre.begin() + hi + 1, mid) - pre.begin();
        if (k > hi) k = hi;
        dep[k] = (int)d;
        stk.push_back({lo, k - 1, d + 1}); stk.push_back({k + 1, hi, d + 1});
    }
    return dep;
}

// ---------------------------------------------------------------- OPT(X) terms
// term_i = log2(min{r>=1 : x_prev_i(n/r) = x_i}) + log2(n / min(x, n-x+1)) for repeat searches.
// x_prev_i(n/r) = x_i  <=>  no other key within n/r of x was searched since x's previous search,
// i.e. n/r < d where d = distance from x to the nearest key searched since then. So the first
// term is log2(max(1, n/d)) (0 when nothing was searched in between).
struct MaxSeg {             // max of last-search time over key ranges, with "nearest key with time > p" queries
    int sz; vector<ll> t;
    void init() { sz = 1; while (sz < n + 2) sz <<= 1; t.assign(2 * sz, -1); }
    void set(int k, ll v) { k += sz; t[k] = v; for (k >>= 1; k; k >>= 1) t[k] = max(t[2 * k], t[2 * k + 1]); }
    // largest key < x with time > p, or -1
    int left(int x, ll p) const { return leftrec(1, 0, sz - 1, x - 1, p); }
    int leftrec(int node, int a, int b, int qhi, ll p) const {
        if (a > qhi || t[node] <= p) return -1;
        if (a == b) return a;
        int m = (a + b) / 2;
        int r = leftrec(2 * node + 1, m + 1, b, qhi, p);
        return r >= 0 ? r : leftrec(2 * node, a, m, qhi, p);
    }
    int right(int x, ll p) const { return rightrec(1, 0, sz - 1, x + 1, p); }
    int rightrec(int node, int a, int b, int qlo, ll p) const {
        if (b < qlo || t[node] <= p) return -1;
        if (a == b) return a;
        int m = (a + b) / 2;
        int r = rightrec(2 * node, a, m, qlo, p);
        return r >= 0 ? r : rightrec(2 * node + 1, m + 1, b, qlo, p);
    }
};

// ---------------------------------------------------------------- workloads
static vector<int> make_workload(const string& w, ll m, uint64_t seed) {
    mt19937_64 rng(seed);
    vector<int> X(m);
    auto uni = [&](ll a, ll b) { return (ll)(a + rng() % (uint64_t)(b - a + 1)); };
    // Zipf(1) sampler over ranks 1..K via inverse CDF table
    auto zipf_table = [&](ll K) { vector<double> c(K); double s = 0; for (ll k = 0; k < K; k++) { s += 1.0 / (k + 1); c[k] = s; } for (auto& v : c) v /= s; return c; };
    auto zipf = [&](const vector<double>& c) { double u = (double)(rng() >> 11) / 9007199254740992.0; return (ll)(lower_bound(c.begin(), c.end(), u) - c.begin()); };
    const ll W = pow3[max(1, L - 4)];   // working-set size for the hot-set workloads (729 at n = 3^10)
    if (w == "uniform") { for (auto& x : X) x = (int)uni(1, n); }
    else if (w == "sequential") { for (ll i = 0; i < m; i++) X[i] = (int)(i % n + 1); }
    else if (w == "walk") {         // random walk, steps of +-1..+-8, reflecting at the ends
        ll x = n / 2;
        for (auto& v : X) { ll s = uni(1, 8) * ((rng() & 1) ? 1 : -1); x += s; if (x < 1) x = 2 - x; if (x > n) x = 2 * n - x; v = (int)x; }
    }
    else if (w == "hot_contig") {  // W adjacent keys, uniform among them, placed at a random offset
        ll base = uni(1, n - W + 1);
        for (auto& x : X) x = (int)(base + uni(0, W - 1));
    }
    else if (w == "hot_spread") {  // W keys chosen at random from all of [n], uniform among them
        vector<int> perm(n); for (ll i = 0; i < n; i++) perm[i] = (int)(i + 1); shuffle(perm.begin(), perm.end(), rng);
        for (auto& x : X) x = perm[uni(0, W - 1)];
    }
    else if (w == "zipf_contig") {  // Zipf over ranks, rank k -> key base+k (popular keys adjacent)
        auto c = zipf_table(n); ll base = uni(1, n);
        for (auto& x : X) x = (int)((base - 1 + zipf(c)) % n + 1);
    }
    else if (w == "zipf_spread") {  // Zipf over ranks, ranks mapped to a random permutation of keys
        auto c = zipf_table(n);
        vector<int> perm(n); for (ll i = 0; i < n; i++) perm[i] = (int)(i + 1); shuffle(perm.begin(), perm.end(), rng);
        for (auto& x : X) x = perm[zipf(c)];
    }
    else if (w.rfind("stride", 0) == 0) {  // strideS: W keys spaced S apart, uniform among them
        ll S = atoll(w.c_str() + 6);
        if (S < 1 || S * (W - 1) >= n) { fprintf(stderr, "stride too large\n"); exit(1); }
        ll base = uni(1, n - S * (W - 1));
        for (auto& x : X) x = (int)(base + S * uni(0, W - 1));
    }
    else { fprintf(stderr, "unknown workload %s\n", w.c_str()); exit(1); }
    return X;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s L m workload seed [check]\n", argv[0]); return 1; }
    L = atoi(argv[1]); ll m = atoll(argv[2]); string w = argv[3]; uint64_t seed = strtoull(argv[4], 0, 10);
    bool check = argc > 5 && string(argv[5]) == "check";
    fill_first = getenv("POPTREE_FILL") && string(getenv("POPTREE_FILL")) == "first";
    pow3.assign(L + 2, 1); for (int i = 1; i <= L + 1; i++) pow3[i] = pow3[i - 1] * 3;
    n = pow3[L];
    vector<int> X = make_workload(w, m, seed);

    SkipPopList sp; sp.init();
    Splay st; st.init();
    MaxSeg seg; seg.init();
    vector<ll> lastSeen(n + 1, -1);
    vector<int> edep = entropy_depth(X);
    double s_ent = 0;
    if (check) sp.check(-1);

    // Measure over the second half only, after every structure has adapted (warm-up = first half).
    ll warm = m / 2;
    double s_pop = 0, s_lvl = 0, s_splay = 0, s_static = 0, s_opt1 = 0, s_opt2 = 0, s_opt = 0;
    ll cnt = 0, cnt_rep = 0;
    ll dem0 = 0, pro0 = 0;
    double min_alpha_seen = 1.0;
    double t_pop = 0, t_splay = 0;
    for (ll i = 0; i < m; i++) {
        ll x = X[i];
        if (i == warm) { dem0 = sp.demotions; pro0 = sp.promotions; }
        int lvl = sp.top[x];
        int cpop = sp.bst_cost(x);
        auto a0 = chrono::steady_clock::now();
        sp.search(x, i);
        auto a1 = chrono::steady_clock::now();
        int cspl = st.access((int)x);
        auto a2 = chrono::steady_clock::now();
        int csta = static_cost(x);
        // OPT terms
        double o1 = 0, o2 = 0; bool rep = lastSeen[x] >= 0;
        if (rep) {
            ll p = lastSeen[x];
            int l = seg.left((int)x, p), r = seg.right((int)x, p);
            ll d = LLONG_MAX;
            if (l >= 1) d = min(d, x - l);
            if (r >= 1 && r <= n) d = min(d, (ll)r - x);
            if (d != LLONG_MAX) o1 = log2(max(1.0, (double)n / d));
            o2 = log2((double)n / min(x, n - x + 1));
        }
        seg.set((int)x, i); lastSeen[x] = i;
        if (check) sp.check(i);
        if (i >= warm) {
            s_pop += cpop; s_ent += edep[x]; s_lvl += lvl; s_splay += cspl; s_static += csta; cnt++;
            t_pop += chrono::duration<double>(a1 - a0).count();
            t_splay += chrono::duration<double>(a2 - a1).count();
            if (rep) { s_opt1 += o1; s_opt2 += o2; s_opt += o1 + o2; cnt_rep++; }
        }
        if (check && (i % 97 == 0 || i == m - 1)) min_alpha_seen = min(min_alpha_seen, sp.min_alpha());
    }
    double ma = check ? min_alpha_seen : sp.min_alpha();
    printf("{\"L\":%d,\"n\":%lld,\"m\":%lld,\"workload\":\"%s\",\"seed\":%llu,"
           "\"poptree\":%.4f,\"poptree_level\":%.4f,\"splay\":%.4f,\"static\":%.4f,\"entropy_static\":%.4f,"
           "\"opt_term1\":%.4f,\"opt_term2\":%.4f,\"opt\":%.4f,\"repeat_frac\":%.4f,"
           "\"restructure_ops\":%.4f,\"min_alpha\":%.5f,\"log2n\":%.4f,"
           "\"ns_poptree_update\":%.1f,\"ns_splay_access\":%.1f,\"checked\":%s}\n",
           L, n, m, w.c_str(), (unsigned long long)seed,
           s_pop / cnt, s_lvl / cnt, s_splay / cnt, s_static / cnt, s_ent / cnt,
           cnt_rep ? s_opt1 / cnt_rep : 0, cnt_rep ? s_opt2 / cnt_rep : 0, cnt_rep ? s_opt / cnt_rep : 0,
           (double)cnt_rep / cnt,
           (double)(sp.demotions - dem0 + sp.promotions - pro0) / cnt, ma, log2((double)n),
           1e9 * t_pop / cnt, 1e9 * t_splay / cnt, check ? "true" : "false");
    return 0;
}
