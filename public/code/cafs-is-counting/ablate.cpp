// Ablation: CAFS with exactly one part swapped. cafs_hashmap keeps CAFS's
// sorted-check, Chao1 sampler, <=8-key branchless path, run merging, fallback
// rule, radix sort of (key,count) pairs and fill-out, but replaces the AVX2
// cache-line bucket table with ankerl::unordered_dense.
#define main bench_main
#include "bench.cpp"
#undef main

template <typename T>
static void cafs_hashmap(std::vector<T>& data) {
    namespace in = cafs2::internal;
    auto fb = [](std::vector<T>& w) { pdqsort(w.begin(), w.end()); };
    const size_t N = data.size();
    if (N < 2) return;
    int order = cafs::detail::check_order(data.data(), N);
    if (order == 1) return;
    if (order == -1) { std::reverse(data.begin(), data.end()); return; }
    if (N < 2048) { fb(data); return; }

    auto sr = in::estimate_k(data.data(), N);
    if (sr.K_est <= 8 && sr.sample_unique <= 8) {  // same key harvest as cafs_sort
        T keys[8]; int n = 0;
        const size_t s = std::min<size_t>(1024, N), stride = std::max<size_t>(1, N / s);
        for (size_t i = 0, t = 0; i < N && t < s && n < 8; i += stride, ++t) {
            bool seen = false;
            for (int j = 0; j < n; ++j) seen |= keys[j] == data[i];
            if (!seen) keys[n++] = data[i];
        }
        for (int i = n; i < 8; ++i) keys[i] = keys[0];
        if (n >= 1 && in::tiny_count_sort_branchless(data, keys, n)) return;
    }
    if (sr.all_unique || sr.K_est * 2 > N) { fb(data); return; }

    ankerl::unordered_dense::map<T, std::uint32_t> m;  // <- the only change
    m.reserve(sr.K_est);
    const T* p = data.data();
    for (size_t i = 0; i < N;) {
        size_t run = 1;
        while (i + run < N && p[i + run] == p[i]) ++run;
        m[p[i]] += std::uint32_t(run);
        i += run;
    }
    auto res = std::move(m).extract();
    in::radix_sort_pairs_by_key(res);
    auto out = data.begin();
    for (auto& [v, c] : res) out = std::fill_n(out, c, v);
}

int main(int argc, char** argv) {
    // usage: ablate <dist> <N> <reps> K1 K2 ...
    std::string dist = argv[1];
    size_t N = std::strtoull(argv[2], nullptr, 10);
    int reps = std::atoi(argv[3]);
    Algo algos[] = {
        {"cafs", [](Vec& v) { cafs2::cafs_sort(v, [](Vec& w) { pdqsort(w.begin(), w.end()); }); }},
        {"cafs_hashmap", [](Vec& v) { cafs_hashmap(v); }},
        {"count_ankerl", count_ankerl},
    };
    for (int a = 4; a < argc; ++a) {
        size_t K = std::strtoull(argv[a], nullptr, 10);
        for (int rep = 0; rep < reps; ++rep) {
            Vec base = make(dist, N, K, 1000 * K + rep);
            Vec ref = base; std::sort(ref.begin(), ref.end());
            for (auto& al : algos) {
                Vec d = base;
                auto t0 = std::chrono::steady_clock::now();
                al.fn(d);
                auto t1 = std::chrono::steady_clock::now();
                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(N);
                std::printf("%s,%zu,%zu,%s,%d,%.4f,%d,%zu\n", dist.c_str(), N, K, al.name, rep, ns,
                            int(d == ref), sizeof(Key) * 8);
                std::fflush(stdout);
            }
        }
    }
}
