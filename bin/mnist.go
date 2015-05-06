package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.001
	maxEpoch  = 50
	learnRate = 0.05
	batchSize = 100
	samples   = 50000
)

// load the data and setup the network with one hidden layer
func setup() (*data.Dataset, *network.Network) {
	fmt.Printf("MNIST DATASET: [784,30,10] layers with cross entropy cost and tanh activation, eta=%.g, batch size=%d\n\n",
		learnRate, batchSize)
	d, err := data.Load("mnist", samples, batchSize)
	checkErr(err)
	net := network.NewNetwork(batchSize)
	net.AddLayer(784, 30, network.NilFunc)
	net.AddLayer(30, 10, network.Tanh)
	net.CrossEntropyOutput(10)
	if debug {
		net.CheckGradient(20, 0.02)
	}
	net.TestBatches(50)
	logEvery = 1
	return d, net
}
