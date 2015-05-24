package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

const hiddenNodes = 30

var cfg = Config{
	TrainRuns: 10,
	MaxEpoch:  50,
	BatchSize: 100,
	LearnRate: 3.0,
	Threshold: 0.0065,
	LogEvery:  1,
}

// load the data and setup the network with one hidden layer
func setup() (*data.Dataset, *network.Network) {
	network.Init(blas.OpenCL32)
	d, err := data.Load("mnist", 0)
	checkErr(err)

	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with quadratic cost and sigmoid activation\n%s\n\n",
		d.NumInputs, hiddenNodes, d.NumOutputs, cfg)

	net := network.New(cfg.BatchSize)
	net.AddLayer(d.NumInputs, hiddenNodes, network.Linear)
	net.AddLayer(hiddenNodes, d.NumOutputs, network.Sigmoid)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	if cfg.Debug {
		for _, l := range net.Nodes[:net.Layers-1] {
			l.Weights().SetFormat("%c")
		}
		net.CheckGradient(1, 1e-6, 8, 1.5)
	}
	return d, net
}
