// Package iris loads default configuration for a network using the iris dataset.
package iris

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/network"
)

func init() {
	network.Register["iris"] = Loader{}
}

type Loader struct{}

func (l Loader) DatasetName() string {
	return "iris"
}

func (l Loader) DefaultConfig() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  200,
		LearnRate: 2.0,
		Threshold: 0.1,
		LogEvery:  25,
		Sampler:   "uniform",
	}
}

func (l Loader) CreateNetwork(cfg *network.Config, d *data.Dataset) *network.Network {
	fmt.Println("IRIS DATASET: single layer with quadratic cost")
	net := network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer(d.NumInputs, d.NumOutputs, network.Linear)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return net
}
