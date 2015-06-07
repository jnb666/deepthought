// Package xor loads default configuration for a network using the xor dataset.
package xor

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/xor"
	"github.com/jnb666/deepthought/network"
)

func init() {
	network.Register["xor"] = Loader{}
}

type Loader struct{}

func (l Loader) DatasetName() string {
	return "xor"
}

func (l Loader) DefaultConfig() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  1000,
		LearnRate: 0.5,
		Threshold: 0.05,
		LogEvery:  25,
		Sampler:   "uniform",
	}
}

func (l Loader) CreateNetwork(cfg *network.Config, d *data.Dataset) *network.Network {
	fmt.Println("XOR DATASET: [2,2,1] layers with quadratic cost and tanh activation")
	net := network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer(2, 2, network.Linear)
	net.AddLayer(2, 1, network.Tanh)
	net.AddQuadraticOutput(1, network.Tanh)
	return net
}
