# one-hash-bloom-vs-split-block

Reproduction code for https://maybenull.com/posts/one-hash-bloom-vs-split-block/

Compares gloom's cache-line one-hash Bloom filter (pinned to commit a1f9124) with
split-block Bloom filters (512-bit and Parquet's 256-bit), a blocked filter with free
positions, and a standard Bloom filter, all at equal memory.

    go test .                         # fastmod == % for every gloom partition
    go run . fpr   -fp 0.01           # false-positive rates      -> fpr.txt
    taskset -c 3 go run . speed -n 1000000      # timings         -> speed.txt
    taskset -c 3 go run . batch -n 100000000    # batched lookups -> batch.txt
    taskset -c 3 go run . lib                   # real gloom.Filter vs SBBF-512 -> lib.txt

Results were produced on a 4-vCPU Intel Xeon @ 2.1 GHz VM with Go 1.26.0.
