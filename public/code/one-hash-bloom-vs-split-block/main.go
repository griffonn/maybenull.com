// Command bloombench compares gloom's one-hash blocked Bloom filter (OHBF) with
// split-block Bloom filters and a standard Bloom filter.
//
//	go run . fpr   [-n 1000000] [-fp 0.01] [-q 10000000] [-seeds 3]
//	go run . speed [-n 1000000] [-fp 0.01] [-q 20000000] [-reps 5]
//	go run . batch [-n 100000000] [-fp 0.01] [-reps 3]
//	go run . lib   [-n 100000000] [-fp 0.01] [-reps 3]
package main

import (
	"fmt"
	"os"
)

var sink int

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: bloombench fpr|speed|batch|lib [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "fpr":
		fprMain(os.Args[2:])
	case "speed":
		speedMain(os.Args[2:], false)
	case "lib":
		libMain(os.Args[2:])
	case "batch":
		speedMain(os.Args[2:], true)
	default:
		fmt.Fprintln(os.Stderr, "unknown mode", os.Args[1])
		os.Exit(2)
	}
}
