package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.05
	maxEpoch  = 200
	learnRate = 2.0
)

// load the data and setup the network - this is a simple single layer net
func setup() (data.Dataset, *network.Network) {
	fmt.Printf("IRIS DATASET: single layer with quadratic cost, eta=%.g\n", learnRate)
	d, err := data.Load("iris", 0)
	checkErr(err)
	net := network.NewNetwork(d.MaxSamples)
	net.AddLayer(d.NumInputs, d.NumOutputs, network.NilFunc)
	net.QuadraticOutput(d.NumOutputs, network.Sigmoid)
	if debug {
		net.CheckGradient(25, 0.02)
	}
	return d, net
}
