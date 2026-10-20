import IPAddress
#if canImport(Glibc)
import Glibc
#endif

// Usage: H check <file>    -> prints "ok a.b.c.d" / "fail" per line
//        H bench <file> <reps>
let args = CommandLine.arguments
let fp = fopen(args[2], "rb")!
var raw: [UInt8] = []
var chunk = [UInt8](repeating: 0, count: 65536)
while true { let n = fread(&chunk, 1, 65536, fp); if n <= 0 { break }; raw += chunk[0..<n] }
let text = String(decoding: raw, as: UTF8.self)
var flat: [UInt8] = []
var ranges: [Range<Int>] = []
// file is binary-safe: one record per line, bytes escaped as \xHH
for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
    let start = flat.count
    var it = Array(line.unicodeScalars).makeIterator()
    while let c = it.next() {
        if c == "\\" { let h = String([Character(it.next()!), Character(it.next()!), Character(it.next()!)].dropFirst()); flat.append(UInt8(h, radix: 16)!) }
        else { flat.append(UInt8(c.value)) }
    }
    ranges.append(start..<flat.count)
}
if ranges.last?.isEmpty == true { ranges.removeLast() }

@inline(never) func parse(_ s: Span<UInt8>) -> IPv4Address? { IPv4Address(textualRepresentation: s) }

if args[1] == "check" {
    let sp = flat.span
    for r in ranges {
        if let a = parse(sp.extracting(r)) { print("ok \(a)") } else { print("fail") }
    }
} else {
    let reps = Int(args[3])!
    let sp = flat.span
    var sink: UInt32 = 0
    var best = Double.infinity
    for _ in 0..<7 {
        var t0 = timespec(); clock_gettime(CLOCK_MONOTONIC, &t0)
        let base = args[1] == "base"
        ranges.withUnsafeBufferPointer { rb in
        for _ in 0..<reps { for i in 0..<rb.count { let r = rb[i]
            if base { sink &+= UInt32(truncatingIfNeeded: sp.extracting(r).count); continue }
            if parse(sp.extracting(r)) != nil { sink &+= 2 } else { sink &+= 1 } } }
        }
        var t1 = timespec(); clock_gettime(CLOCK_MONOTONIC, &t1)
        let ns = Double(t1.tv_sec - t0.tv_sec) * 1e9 + Double(t1.tv_nsec - t0.tv_nsec)
        best = min(best, ns / Double(reps * ranges.count))
    }
    print("\(Double(Int(best * 100)) / 100) ns/op (sink \(sink))")
}
