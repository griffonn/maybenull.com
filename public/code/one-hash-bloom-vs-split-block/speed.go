package main

import (
	"flag"
	"fmt"
	"sort"
	"time"

	"github.com/jcalabro/gloom"
)

// splitmix64 finalizer: stands in for xxh3 output so no memory traffic is spent on keys.
func mix(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ x>>30) * 0xBF58476D1CE4E5B9
	x = (x ^ x>>27) * 0x94D049BB133111EB
	return x ^ x>>31
}

type res struct{ add, neg, pos []float64 }

func med(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func speedMain(args []string, batched bool) {
	fs := flag.NewFlagSet("speed", flag.ExitOnError)
	n := fs.Uint64("n", 1_000_000, "items")
	fp := fs.Float64("fp", 0.01, "")
	q := fs.Uint64("q", 20_000_000, "")
	reps := fs.Int("reps", 5, "")
	fs.Parse(args)
	blocks, k, _ := gloom.OptimalParams(*n, *fp)
	M := blocks * 512
	fmt.Printf("n=%d filter=%.1f MB k=%d\n", *n, float64(M)/8/1e6, k)
	names := []string{"OHBF (gloom scheme)", "OHBF + fastmod", "blocked512-k", "SBBF-512 k8", "SBBF-256 k8", "standard KM"}
	if batched {
		for r := 0; r < *reps; r++ {
			runBatch("OHBF (gloom scheme)", NewOH(blocks, k, false), *n, *q)
			runBatch("OHBF + fastmod", NewOHF(blocks, k), *n, *q)
			runBatch("SBBF-512 k8", NewSBBF512(blocks), *n, *q)
		}
		return
	}
	R := make([]res, len(names))
	for rep := 0; rep < *reps; rep++ {
		for j := range names {
			var a, ng, p float64
			switch j {
			case 0:
				a, ng, p = runOH(NewOH(blocks, k, false), *n, *q)
			case 1:
				a, ng, p = runOHF(NewOHF(blocks, k), *n, *q)
			case 2:
				a, ng, p = runBlk(NewBlk(blocks, int(k)), *n, *q)
			case 3:
				a, ng, p = runSBBF512(NewSBBF512(blocks), *n, *q)
			case 4:
				a, ng, p = runSBBF256(NewSBBF256(M), *n, *q)
			case 5:
				a, ng, p = runStd(NewStd(M, *n), *n, *q)
			}
			R[j].add = append(R[j].add, a)
			R[j].neg = append(R[j].neg, ng)
			R[j].pos = append(R[j].pos, p)
		}
	}
	fmt.Printf("  %-22s %8s %8s %8s   (median of %d, ns/op)\n", "", "add", "miss", "hit", *reps)
	for j, nm := range names {
		fmt.Printf("  %-22s %8.2f %8.2f %8.2f\n", nm, med(R[j].add), med(R[j].neg), med(R[j].pos))
	}
}
func runOH(f *OH, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
func runOHF(f *OHF, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
func runBlk(f *Blk, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
func runSBBF512(f *SBBF512, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
func runSBBF256(f *SBBF256, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
func runStd(f *Std, n, q uint64) (float64, float64, float64) {
	t := time.Now()
	for i := uint64(0); i < n; i++ {
		f.Add(mix(i), mix(i^0x5555555555555555))
	}
	add := float64(time.Since(t).Nanoseconds()) / float64(n)
	c := 0
	t = time.Now()
	for i := uint64(1 << 40); i < 1<<40+q; i++ {
		if f.Has(mix(i), mix(i^0x5555555555555555)) {
			c++
		}
	}
	neg := float64(time.Since(t).Nanoseconds()) / float64(q)
	t = time.Now()
	for i := uint64(0); i < q; i++ {
		j := i % n
		if f.Has(mix(j), mix(j^0x5555555555555555)) {
			c++
		}
	}
	pos := float64(time.Since(t).Nanoseconds()) / float64(q)
	sink += c
	return add, neg, pos
}
