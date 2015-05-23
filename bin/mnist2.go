package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.0
	stopAfter = 3
	maxEpoch  = 50
	learnRate = 0.5
	batchSize = 100
	samples   = 50000
	nHidden   = 100
)

// load the data and setup the network with one hidden layer
func setup() (*data.Dataset, *network.Network) {
	network.Init(blas.OpenCL32)
	d, err := data.Load("mnist", samples, batchSize)
	checkErr(err)

	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with cross entropy cost and sigmoid activation, eta=%.g, batch size=%d\n\n",
		d.NumInputs, nHidden, d.NumOutputs, learnRate, batchSize)

	net := network.NewNetwork(batchSize)
	net.AddLayer(d.NumInputs, nHidden, network.Linear)
	net.AddLayer(nHidden, d.NumOutputs, network.Sigmoid)
	net.CrossEntropyOutput(d.NumOutputs)
	if debug {
		for _, l := range net.Nodes[:net.Layers-1] {
			l.Weights().SetFormat("%c")
		}
		net.CheckGradient(1, 1e-6, 8, 1.5)
	}
	net.TestBatches(100)
	logEvery = 1
	return d, net
}
