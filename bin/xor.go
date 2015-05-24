package main

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/xor"
	"github.com/jnb666/deepthought/network"
)

var cfg = Config{
	TrainRuns: 30,
	MaxEpoch:  1000,
	LearnRate: 0.5,
	Threshold: 0.05,
}

// load the data and setup the network with one hidden layer
func setup() (*data.Dataset, *network.Network) {
	network.Init(blas.Native64)
	fmt.Printf("XOR DATASET: [2,2,1] layers with quadratic cost and tanh activation\n%s\n", cfg)
	d, err := data.Load("xor", 0)
	checkErr(err)
	if cfg.Debug {
		fmt.Println(d.Train)
	}
	net := network.New(d.MaxSamples)
	net.AddLayer(2, 2, network.Linear)
	net.AddLayer(2, 1, network.Tanh)
	net.AddQuadraticOutput(1, network.Tanh)
	if cfg.Debug {
		net.CheckGradient(20, 1e-6, 0, 0.25)
	}
	return d, net
}
