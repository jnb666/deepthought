package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

const hiddenNodes = 400

var cfg = Config{
	TrainRuns: 5,
	MaxEpoch:  100,
	BatchSize: 250,
	LearnRate: 0.5,
	Momentum:  0.7,
	StopAfter: 8,
	LogEvery:  5,
}

// load the data and setup the network with one hidden layer
func setup() (*data.Dataset, *network.Network) {
	network.Init(blas.OpenCL32)
	d, err := data.Load("mnist", 0)
	checkErr(err)

	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with cross entropy cost and relu activation\n%s\n\n",
		d.NumInputs, hiddenNodes, d.NumOutputs, cfg)

	net := network.New(cfg.BatchSize)
	net.AddLayer(d.NumInputs, hiddenNodes, network.Linear)
	net.AddLayer(hiddenNodes, d.NumOutputs, network.Relu)
	net.AddCrossEntropyOutput(d.NumOutputs)
	return d, net
}
