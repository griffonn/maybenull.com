// CAFS vs. counting-sort baselines on low-cardinality uint64 arrays.
// Build: see README.md. Output: CSV on stdout.
#include "third_party/cafs2.hpp"
#include "third_party/pdqsort.h"
#include "third_party/ska_sort.hpp"
#include "third_party/unordered_dense.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using u64 = std::uint64_t;
#ifdef KEY32
using Key = std::int32_t;  // CAFS's AVX2 bucket holds 8 of these per cache line
#else
using Key = u64;           // ... and 4 of these
#endif
using Vec = std::vector<Key>;

// ---- counting baselines: count, sort the K distinct keys, write runs back ----

template <class Pairs>
static void emit(Vec& v, Pairs& kv) {
    std::sort(kv.begin(), kv.end());
    auto out = v.begin();
    for (auto& [k, c] : kv) out = std::fill_n(out, c, k);
}

static void count_std_umap(Vec& v) {
    std::unordered_map<Key, u64> m;
    for (Key x : v) ++m[x];
    std::vector<std::pair<Key, u64>> kv(m.begin(), m.end());
    emit(v, kv);
}

static void count_ankerl(Vec& v) {
    ankerl::unordered_dense::map<Key, u64> m;
    for (Key x : v) ++m[x];
    auto kv = std::move(m).extract();  // already a vector<pair>
    emit(v, kv);
}

// ~30 lines: linear probing over {key,count} slots, Fibonacci hash, doubles at 50% load.
static void count_flat(Vec& v) {
    struct Slot { Key key; u64 cnt; };
    int shift = 64 - 6;
    size_t mask = 63, used = 0;
    std::vector<Slot> t(mask + 1);
    Slot* T = t.data();
    for (Key x : v) {
        size_t i = (u64(std::make_unsigned_t<Key>(x)) * 11400714819323198485ull) >> shift;
        while (T[i].cnt && T[i].key != x) i = (i + 1) & mask;
        if (T[i].cnt++ == 0) {
            T[i].key = x;
            if (++used * 2 > mask + 1) {  // grow and rehash
                std::vector<Slot> old(2 * (mask + 1));
                old.swap(t);
                T = t.data(); mask = t.size() - 1; --shift;
                for (auto& o : old) if (o.cnt) {
                    size_t j = (u64(std::make_unsigned_t<Key>(o.key)) * 11400714819323198485ull) >> shift;
                    while (T[j].cnt) j = (j + 1) & mask;
                    T[j] = o;
                }
            }
        }
    }
    std::vector<std::pair<Key, u64>> kv;
    kv.reserve(used);
    for (auto& s : t) if (s.cnt) kv.emplace_back(s.key, s.cnt);
    emit(v, kv);
}

// ---- inputs ----

// K distinct keys spread over the 64-bit range: splitmix64 finalizer, a bijection.
// (Not a golden-ratio multiple: CAFS and count_flat both hash with one, and the
// two would interact.)
static Key key_of(u64 i) {
    u64 z = i + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return Key(z ^ (z >> 31));  // KEY32: low 32 bits (a few collisions at K=1e6)
}

static Vec make(const std::string& dist, size_t N, size_t K, u64 seed) {
    std::mt19937_64 r(seed);
    Vec v(N);
    if (dist == "uniform") {
        std::uniform_int_distribution<u64> d(0, K - 1);
        for (auto& x : v) x = key_of(d(r));
    } else if (dist == "zipf") {  // Zipf s=1 over K keys, inverse-CDF sampling
        std::vector<double> cdf(K);
        double s = 0;
        for (size_t i = 0; i < K; ++i) cdf[i] = (s += 1.0 / double(i + 1));
        std::uniform_real_distribution<double> u(0, s);
        for (auto& x : v) x = key_of(std::lower_bound(cdf.begin(), cdf.end(), u(r)) - cdf.begin());
    } else if (dist == "dense") {  // keys 0..K-1 (small range, the counting-array case)
        std::uniform_int_distribution<u64> d(0, K - 1);
        for (auto& x : v) x = Key(d(r));
    } else if (dist == "runs") {  // uniform keys in runs of geometric length, mean 16 (clustered column)
        std::uniform_int_distribution<u64> d(0, K - 1);
        std::geometric_distribution<size_t> len(1.0 / 16);
        for (size_t i = 0; i < N;) {
            Key k = key_of(d(r));
            for (size_t n = len(r) + 1; n-- && i < N;) v[i++] = k;
        }
    } else {
        std::fprintf(stderr, "unknown dist %s\n", dist.c_str()); std::exit(1);
    }
    return v;
}

struct Algo { const char* name; void (*fn)(Vec&); };

int main(int argc, char** argv) {
    // usage: bench <dist> <N> <reps> K1 K2 ...
    std::string dist = argv[1];
    size_t N = std::strtoull(argv[2], nullptr, 10);
    int reps = std::atoi(argv[3]);
    Algo algos[] = {
        {"cafs", [](Vec& v) { cafs2::cafs_sort(v, [](Vec& w) { pdqsort(w.begin(), w.end()); }); }},
        {"std_sort", [](Vec& v) { std::sort(v.begin(), v.end()); }},
        {"pdqsort", [](Vec& v) { pdqsort(v.begin(), v.end()); }},
        {"ska_sort", [](Vec& v) { ska_sort(v.begin(), v.end()); }},
        {"count_std_umap", count_std_umap},
        {"count_ankerl", count_ankerl},
        {"count_flat", count_flat},
    };
    for (int a = 4; a < argc; ++a) {
        size_t K = std::strtoull(argv[a], nullptr, 10);
        Vec ref;
        for (int rep = 0; rep < reps; ++rep) {
            Vec base = make(dist, N, K, 1000 * K + rep);
            if (rep == 0) { ref = base; std::sort(ref.begin(), ref.end()); }
            for (auto& al : algos) {
                Vec d = base;
                auto t0 = std::chrono::steady_clock::now();
                al.fn(d);
                auto t1 = std::chrono::steady_clock::now();
                bool ok = rep ? std::is_sorted(d.begin(), d.end()) : d == ref;
                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(N);
                std::printf("%s,%zu,%zu,%s,%d,%.4f,%d,%zu\n", dist.c_str(), N, K, al.name, rep, ns, ok, sizeof(Key) * 8);
                std::fflush(stdout);
            }
        }
    }
}
