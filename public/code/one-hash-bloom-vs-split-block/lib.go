package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"time"

	"github.com/jcalabro/gloom"
	"github.com/zeebo/xxh3"
)

// libMain times the real gloom.Filter (which hashes keys with xxh3 itself) against
// SBBF-512 fed by the same xxh3 hash, on 8-byte keys.
func libMain(args []string) {
	fs := flag.NewFlagSet("lib", flag.ExitOnError)
	n := fs.Uint64("n", 100_000_000, "items")
	fp := fs.Float64("fp", 0.01, "")
	q := fs.Int("q", 20_000_000, "")
	reps := fs.Int("reps", 3, "")
	fs.Parse(args)
	blocks, k, _ := gloom.OptimalParams(*n, *fp)
	fmt.Printf("n=%d filter=%.1f MB k=%d\n", *n, float64(blocks*512)/8/1e6, k)
	for r := 0; r < *reps; r++ {
		g := gloom.NewWithParams(blocks, k)
		s := NewSBBF512(blocks)
		buf := make([]byte, 8)
		c := 0
		t := time.Now()
		for i := uint64(0); i < *n; i++ {
			binary.LittleEndian.PutUint64(buf, i)
			g.Add(buf)
		}
		ga := float64(time.Since(t).Nanoseconds()) / float64(*n)
		t = time.Now()
		for i := uint64(0); i < *n; i++ {
			binary.LittleEndian.PutUint64(buf, i)
			h := xxh3.Hash128(buf)
			s.Add(h.Hi, h.Lo)
		}
		sa := float64(time.Since(t).Nanoseconds()) / float64(*n)
		t = time.Now()
		for i := 0; i < *q; i++ {
			binary.LittleEndian.PutUint64(buf, uint64(i)|1<<60)
			if g.Test(buf) {
				c++
			}
		}
		gm := float64(time.Since(t).Nanoseconds()) / float64(*q)
		t = time.Now()
		for i := 0; i < *q; i++ {
			binary.LittleEndian.PutUint64(buf, uint64(i)|1<<60)
			h := xxh3.Hash128(buf)
			if s.Has(h.Hi, h.Lo) {
				c++
			}
		}
		sm := float64(time.Since(t).Nanoseconds()) / float64(*q)
		t = time.Now()
		for i := 0; i < *q; i++ {
			binary.LittleEndian.PutUint64(buf, uint64(i)%*n)
			if g.Test(buf) {
				c++
			}
		}
		gh := float64(time.Since(t).Nanoseconds()) / float64(*q)
		t = time.Now()
		for i := 0; i < *q; i++ {
			binary.LittleEndian.PutUint64(buf, uint64(i)%*n)
			h := xxh3.Hash128(buf)
			if s.Has(h.Hi, h.Lo) {
				c++
			}
		}
		sh := float64(time.Since(t).Nanoseconds()) / float64(*q)
		sink += c
		fmt.Printf("  gloom.Filter     add %7.2f  miss %7.2f  hit %7.2f ns/op\n", ga, gm, gh)
		fmt.Printf("  SBBF-512 + xxh3  add %7.2f  miss %7.2f  hit %7.2f ns/op\n", sa, sm, sh)
	}
}
