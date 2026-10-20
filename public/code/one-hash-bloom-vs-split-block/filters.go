package main

import (
	"math"
	"math/bits"

	"github.com/jcalabro/gloom"
)

func reduce(h, n uint64) uint64 { hi, _ := bits.Mul64(h, n); return hi }

// ---- Standard (unblocked) Bloom, Kirsch-Mitzenmacher double hashing ----
type Std struct {
	w []uint64
	m uint64
	k uint64
}

func NewStd(mbits uint64, n uint64) *Std {
	k := uint64(math.Round(float64(mbits) / float64(n) * math.Ln2))
	return &Std{w: make([]uint64, (mbits+63)/64), m: mbits, k: k}
}
func (f *Std) Add(hi, lo uint64) {
	for i := uint64(0); i < f.k; i++ {
		p := reduce(hi+i*lo, f.m)
		f.w[p>>6] |= 1 << (p & 63)
	}
}
func (f *Std) Has(hi, lo uint64) bool {
	for i := uint64(0); i < f.k; i++ {
		p := reduce(hi+i*lo, f.m)
		if f.w[p>>6]&(1<<(p&63)) == 0 {
			return false
		}
	}
	return true
}

// ---- Blocked 512, k positions anywhere in the block (9 bits each from lo, then remix) ----
type Blk struct {
	w []uint64
	b uint64
	k int
}

func NewBlk(blocks uint64, k int) *Blk { return &Blk{w: make([]uint64, blocks*8), b: blocks, k: k} }
func (f *Blk) Add(hi, lo uint64) {
	base := reduce(hi, f.b) * 8
	h := lo
	for i := 0; i < f.k; i++ {
		if i == 7 {
			h = lo*0x9E3779B97F4A7C15 ^ hi
		}
		p := h & 511
		h >>= 9
		f.w[base+p>>6] |= 1 << (p & 63)
	}
}
func (f *Blk) Has(hi, lo uint64) bool {
	base := reduce(hi, f.b) * 8
	h := lo
	for i := 0; i < f.k; i++ {
		if i == 7 {
			h = lo*0x9E3779B97F4A7C15 ^ hi
		}
		p := h & 511
		h >>= 9
		if f.w[base+p>>6]&(1<<(p&63)) == 0 {
			return false
		}
	}
	return true
}

// ---- Split block Bloom filter, Parquet spec: 256-bit blocks, 8 x 32-bit words, 1 bit/word ----
var salt = [8]uint32{0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31}

type SBBF256 struct {
	w []uint32
	b uint64
}

func NewSBBF256(mbits uint64) *SBBF256 {
	b := mbits / 256
	return &SBBF256{w: make([]uint32, b*8), b: b}
}
func (f *SBBF256) Add(hi, lo uint64) {
	blk := f.w[reduce(hi, f.b)*8:][:8]
	x := uint32(lo)
	for i := 0; i < 8; i++ {
		blk[i] |= 1 << ((x * salt[i]) >> 27)
	}
}
func (f *SBBF256) Has(hi, lo uint64) bool {
	blk := f.w[reduce(hi, f.b)*8:][:8]
	x := uint32(lo)
	for i := 0; i < 8; i++ {
		if blk[i]&(1<<((x*salt[i])>>27)) == 0 {
			return false
		}
	}
	return true
}

// ---- Split block, 512-bit blocks (one cache line), 8 x 64-bit words, 1 bit/word ----
type SBBF512 struct {
	w []uint64
	b uint64
}

func NewSBBF512(blocks uint64) *SBBF512 { return &SBBF512{w: make([]uint64, blocks*8), b: blocks} }
func (f *SBBF512) Add(hi, lo uint64) {
	blk := f.w[reduce(hi, f.b)*8:][:8]
	x := uint32(lo)
	for i := 0; i < 8; i++ {
		blk[i] |= 1 << ((x * salt[i]) >> 26)
	}
}
func (f *SBBF512) Has(hi, lo uint64) bool {
	blk := f.w[reduce(hi, f.b)*8:][:8]
	x := uint32(lo)
	for i := 0; i < 8; i++ {
		if blk[i]&(1<<((x*salt[i])>>26)) == 0 {
			return false
		}
	}
	return true
}

// ---- OHBF reimplementation, intra-hash either 32-bit fold (as gloom) or full 64-bit ----
type OH struct {
	w       []uint64
	b       uint64
	primes  []uint32
	offsets []uint32
	wide    bool
}

func NewOH(blocks uint64, k uint32, wide bool) *OH {
	p := gloom.GetPrimePartition(k)
	return &OH{w: make([]uint64, blocks*8), b: blocks, primes: p, offsets: gloom.ComputeOffsets(p), wide: wide}
}
func (f *OH) pos(i int, lo uint64) uint32 {
	if f.wide {
		return f.offsets[i] + uint32(lo%uint64(f.primes[i]))
	}
	return f.offsets[i] + (uint32(lo)^uint32(lo>>32))%f.primes[i]
}
func (f *OH) Add(hi, lo uint64) {
	base := reduce(hi, f.b) * 8
	for i := range f.primes {
		p := f.pos(i, lo)
		f.w[base+uint64(p>>6)] |= 1 << (p & 63)
	}
}
func (f *OH) Has(hi, lo uint64) bool {
	base := reduce(hi, f.b) * 8
	for i := range f.primes {
		p := f.pos(i, lo)
		if f.w[base+uint64(p>>6)]&(1<<(p&63)) == 0 {
			return false
		}
	}
	return true
}

// ---- OHBF with Lemire fastmod (precomputed per-partition magic constant, no DIV) ----
type OHF struct {
	w       []uint64
	b       uint64
	primes  []uint32
	offsets []uint32
	magic   []uint64
}

func NewOHF(blocks uint64, k uint32) *OHF {
	p := gloom.GetPrimePartition(k)
	m := make([]uint64, len(p))
	for i, d := range p {
		m[i] = ^uint64(0)/uint64(d) + 1
	}
	return &OHF{w: make([]uint64, blocks*8), b: blocks, primes: p, offsets: gloom.ComputeOffsets(p), magic: m}
}
func fastmod(a uint32, M uint64, d uint32) uint32 {
	hi, _ := bits.Mul64(M*uint64(a), uint64(d))
	return uint32(hi)
}
func (f *OHF) Add(hi, lo uint64) {
	base := reduce(hi, f.b) * 8
	x := uint32(lo) ^ uint32(lo>>32)
	for i := range f.primes {
		p := f.offsets[i] + fastmod(x, f.magic[i], f.primes[i])
		f.w[base+uint64(p>>6)] |= 1 << (p & 63)
	}
}
func (f *OHF) Has(hi, lo uint64) bool {
	base := reduce(hi, f.b) * 8
	x := uint32(lo) ^ uint32(lo>>32)
	for i := range f.primes {
		p := f.offsets[i] + fastmod(x, f.magic[i], f.primes[i])
		if f.w[base+uint64(p>>6)]&(1<<(p&63)) == 0 {
			return false
		}
	}
	return true
}

// Batched lookups: first touch every key's block (issuing all cache misses back to
// back), then run the normal probes, which now hit in cache.
const batch = 16

var touchSink uint64

func (f *OHF) HasBatch(his, los *[batch]uint64, out *[batch]bool) {
	var t uint64
	for i := 0; i < batch; i++ {
		t |= f.w[reduce(his[i], f.b)*8]
	}
	touchSink += t
	for i := 0; i < batch; i++ {
		out[i] = f.Has(his[i], los[i])
	}
}
func (f *SBBF512) HasBatch(his, los *[batch]uint64, out *[batch]bool) {
	var t uint64
	for i := 0; i < batch; i++ {
		t |= f.w[reduce(his[i], f.b)*8]
	}
	touchSink += t
	for i := 0; i < batch; i++ {
		out[i] = f.Has(his[i], los[i])
	}
}
func (f *OH) HasBatch(his, los *[batch]uint64, out *[batch]bool) {
	var t uint64
	for i := 0; i < batch; i++ {
		t |= f.w[reduce(his[i], f.b)*8]
	}
	touchSink += t
	for i := 0; i < batch; i++ {
		out[i] = f.Has(his[i], los[i])
	}
}
