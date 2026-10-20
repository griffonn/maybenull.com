package main

import (
	"math/rand/v2"
	"testing"
)

func TestFastmod(t *testing.T) {
	for k := uint32(3); k <= 17; k++ {
		a, b := NewOH(64, k, false), NewOHF(64, k)
		for r := 0; r < 5000; r++ {
			a, b = NewOH(64, k, false), NewOHF(64, k)
			for i := 0; i < 300; i++ {
				hi, lo := rand.Uint64(), rand.Uint64()
				a.Add(hi, lo)
				b.Add(hi, lo)
				for j := range a.w {
					if a.w[j] != b.w[j] {
						t.Fatalf("k=%d mismatch", k)
					}
				}
			}
		}
		for _, d := range b.primes {
			M := ^uint64(0)/uint64(d) + 1
			for _, x := range []uint32{0, 1, d - 1, d, ^uint32(0), ^uint32(0) - 1} {
				if fastmod(x, M, d) != x%d {
					t.Fatal(x, d)
				}
			}
		}
	}
}
