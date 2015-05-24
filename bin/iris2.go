package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/network"
)

var cfg = Config{
	TrainRuns: 30,
	MaxEpoch:  100,
	LearnRate: 1.0,
	Threshold: 0.15,
}

// load the data and setup the network - this is a simple single layer net
func setup() (*data.Dataset, *network.Network) {
	network.Init(blas.Native64)
	fmt.Printf("IRIS DATASET: single layer with cross entropy cost\n%s\n", cfg)
	d, err := data.Load("iris", 0)
	checkErr(err)
	net := network.New(d.MaxSamples)
	net.AddLayer(d.NumInputs, d.NumOutputs, network.Linear)
	net.AddCrossEntropyOutput(d.NumOutputs)
	if cfg.Debug {
		net.CheckGradient(20, 0.001, 8, 1)
	}
	return d, net
}
