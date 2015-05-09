package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.15
	maxEpoch  = 100
	learnRate = 1.0
)

// load the data and setup the network - this is a simple single layer net
func setup() (*data.Dataset, *network.Network) {
	fmt.Printf("IRIS DATASET: single layer with cross entropy cost, eta=%.g\n", learnRate)
	d, err := data.Load("iris", 0, 0)
	checkErr(err)
	net := network.NewNetwork(d.MaxSamples)
	net.AddLayer(d.NumInputs, d.NumOutputs, network.Linear)
	net.CrossEntropyOutput(d.NumOutputs)
	if debug {
		net.CheckGradient(20, 1e-6, 0, 1)
	}
	return d, net
}
