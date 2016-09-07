package main

import "github.com/moverest/neuralnet"

func maxFloatlSliceValueIndex(s []float64) int {
	if len(s) == 0 {
		return 0
	}

	maxI := 0
	for i := range s {
		if s[i] > s[maxI] {
			maxI = i
		}
	}

	return maxI
}

// Evaluate return the number of successful guesses made by the network.
func Evaluate(net *neuralnet.Network, test neuralnet.Set) int {
	nCorrects := 0
	for i := 0; i < test.Count(); i++ {
		in, expectedOut := test.GetVects(i)
		out := net.FeedForward(in)
		if maxFloatlSliceValueIndex(out) == maxFloatlSliceValueIndex(expectedOut) {
			nCorrects++
		}
	}

	return nCorrects
}
