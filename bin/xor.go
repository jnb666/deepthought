package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/xor"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.05
	maxEpoch  = 1000
	learnRate = 0.5
)

// load the data and setup the network with one hidden layer
func setup() (data.Dataset, *network.Network) {
	fmt.Printf("XOR DATASET: [2,2,1] layers with quadratic cost and tanh activation, eta=%.g\n\n", learnRate)
	d, err := data.Load("xor", 0)
	checkErr(err)
	net := network.NewNetwork(d.MaxSamples)
	net.AddLayer(2, 2, network.NilFunc)
	net.AddLayer(2, 1, network.Tanh)
	net.QuadraticOutput(1, network.Tanh)
	if debug {
		net.CheckGradient(20, 0.02)
	}
	return d, net
}
