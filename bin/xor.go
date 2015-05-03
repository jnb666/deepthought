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
	fmt.Printf("XOR DATASET: [2,2,1] layers with cross entropy cost and tanh activation, eta=%.g\n", learnRate)
	d, err := data.Load("xor", 0)
	checkErr(err)
	net := network.NewNetwork(d.MaxSamples)
	net.InputLayer(2, 2)
	net.HiddenLayer(2, 1, network.TanhActivation)
	net.QuadraticOutput(1, network.TanhActivation)
	if debug {
		net.CheckGradient(20, 0.02)
	}
	return d, net
}
