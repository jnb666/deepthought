package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

var (
	threshold = 0.001
	maxEpoch  = 100
	learnRate = 3.0
	batchSize = 1000
)

// load the data and setup the network with one hidden layer
func setup() (data.Dataset, *network.Network) {
	fmt.Printf("MNIST DATASET: [784,30,10] layers with quadratic cost and sigmoid activation, eta=%.g\n\n", learnRate)
	d, err := data.Load("mnist", batchSize)
	checkErr(err)
	net := network.NewNetwork(batchSize)
	net.AddLayer(784, 30, network.NilFunc)
	net.AddLayer(30, 10, network.Sigmoid)
	net.QuadraticOutput(10, network.Sigmoid)
	if debug {
		net.CheckGradient(20, 0.02)
	}
	return d, net
}
