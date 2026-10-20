package main

import (
	"fmt"
	"time"
)

type batcher interface {
	Add(hi, lo uint64)
	HasBatch(his, los *[batch]uint64, out *[batch]bool)
}

func runBatch(name string, f batcher, n, q uint64) {
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	var his, los [batch]uint64
	var out [batch]bool
	c := 0
	t := time.Now()
	for i := uint64(0); i < q; i += batch {
		for j := uint64(0); j < batch; j++ {
			k := (i + j) % n
			his[j], los[j] = mix(k), mix(k^0x5555555555555555)
		}
		f.HasBatch(&his, &los, &out)
		for _, b := range out {
			if b {
				c++
			}
		}
	}
	hit := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i += batch {
		for j := uint64(0); j < batch; j++ {
			k := i + j
			his[j], los[j] = mix(k), mix(k^0x5555555555555555)
		}
		f.HasBatch(&his, &los, &out)
		for _, b := range out {
			if b {
				c++
			}
		}
	}
	miss := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	fmt.Printf("  batched %-20s miss %7.2f  hit %7.2f ns/op\n", name, miss, hit)
}
