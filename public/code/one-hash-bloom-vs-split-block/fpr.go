package main

import (
	"encoding/binary"
	"flag"
	"fmt"

	"github.com/jcalabro/gloom"
	"github.com/zeebo/xxh3"
)

type filter interface {
	Add(hi, lo uint64)
	Has(hi, lo uint64) bool
}

func key(buf []byte, seed, i uint64) []byte {
	binary.LittleEndian.PutUint64(buf, seed<<40|i)
	return buf
}

func fprMain(args []string) {
	fs := flag.NewFlagSet("fpr", flag.ExitOnError)
	n := fs.Uint64("n", 1_000_000, "items")
	fp := fs.Float64("fp", 0.01, "target")
	q := fs.Uint64("q", 10_000_000, "queries")
	seeds := fs.Int("seeds", 3, "")
	fs.Parse(args)
	blocks, k, _ := gloom.OptimalParams(*n, *fp)
	M := blocks * 512
	fmt.Printf("n=%d target=%g blocks=%d k=%d bits/item=%.2f gloomEst=%.4f%%\n", *n, *fp, blocks, k, float64(M)/float64(*n), 100*gloom.EstimateFalsePositiveRate(blocks, k, *n))
	names := []string{"gloom(lib)", "OHBF-32bit", "OHBF-64bit", "blocked512-k", "SBBF-512 k8", "SBBF-256 k8 (Parquet)", "standard KM"}
	tot := make([]uint64, len(names))
	buf := make([]byte, 8)
	for s := 1; s <= *seeds; s++ {
		g := gloom.NewWithParams(blocks, k)
		fs := []filter{NewOH(blocks, k, false), NewOH(blocks, k, true), NewBlk(blocks, int(k)), NewSBBF512(blocks), NewSBBF256(M), NewStd(M, *n)}
		for i := uint64(0); i < *n; i++ {
			kb := key(buf, uint64(s), i)
			g.Add(kb)
			h := xxh3.Hash128(kb)
			for _, f := range fs {
				f.Add(h.Hi, h.Lo)
			}
		}
		for i := uint64(0); i < *q; i++ {
			kb := key(buf, uint64(s)+1000, i)
			if g.Test(kb) {
				tot[0]++
			}
			h := xxh3.Hash128(kb)
			for j, f := range fs {
				if f.Has(h.Hi, h.Lo) {
					tot[j+1]++
				}
			}
		}
	}
	Q := float64(*q) * float64(*seeds)
	for j, nm := range names {
		fmt.Printf("  %-24s FPR=%.4f%%  (%d fps)\n", nm, 100*float64(tot[j])/Q, tot[j])
	}
}
