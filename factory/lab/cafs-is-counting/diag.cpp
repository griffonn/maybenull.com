// Which path does CAFS take, and how big is its table? Re-runs CAFS's own
// estimator and sizing (cafs2::internal), then replays its bucket loop to count spills.
#include "third_party/cafs2.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using Vec = std::vector<u64>;
#define main bench_main
#include "bench.cpp"
#undef main

int main(int argc, char** argv) {
    std::string dist = argv[1];
    size_t N = std::strtoull(argv[2], nullptr, 10);
    std::printf("dist,N,K,true_K,K_est,path,buckets,table_bytes,spilled_elems\n");
    for (int a = 3; a < argc; ++a) {
        size_t K = std::strtoull(argv[a], nullptr, 10);
        Vec v = make(dist, N, K, 1000 * K);
        Vec s = v; std::sort(s.begin(), s.end());
        size_t trueK = std::unique(s.begin(), s.end()) - s.begin();
        auto sr = cafs2::internal::estimate_k(v.data(), N);
        const char* path = "buckets";
        size_t nb = 0, spilled = 0;
        if (sr.K_est <= 8 && sr.sample_unique <= 8) path = "tiny8";
        else if (sr.all_unique || sr.K_est * 2 > N) path = "fallback";
        if (std::string(path) != "fallback") {  // tiny8 falls through to buckets if a key was missed
            nb = cafs2::internal::pick_num_buckets<u64>(sr.K_est, N);
            int shift = 64 - std::bit_width(nb - 1);
            std::vector<cafs::detail::Bucket<u64>> table(nb);
            for (size_t i = 0; i < N; ++i) {
                size_t idx = cafs::detail::fast_hash(v[i], shift) & (nb - 1);
                if (!table[idx].update(v[i], 1)) ++spilled;
            }
            if (std::string(path) == "tiny8" && trueK > 8) path = "tiny8_missed->buckets";
            else if (std::string(path) == "tiny8") nb = 0, spilled = 0;
        }
        if (spilled > N / 2) path = "buckets->fallback";
        std::printf("%s,%zu,%zu,%zu,%zu,%s,%zu,%zu,%zu\n", dist.c_str(), N, K, trueK, sr.K_est,
                    path, nb, nb * sizeof(cafs::detail::Bucket<u64>), spilled);
    }
}
