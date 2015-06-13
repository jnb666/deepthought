// Package mnist loads default configuration for a network using the mnist dataset.
package mnist

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
	"math"
)

func init() {
	network.Register["mnist"] = Loader{}
	network.Register["mnist2"] = Loader2{}
}

type Loader struct{}

func (l Loader) DatasetName() string {
	return "mnist"
}

func (l Loader) DefaultConfig() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  50,
		BatchSize: 100,
		LearnRate: 3.0,
		Threshold: 0.0067,
		LogEvery:  1,
		Sampler:   "random",
	}
}

func (l Loader) CreateNetwork(cfg *network.Config, d *data.Dataset) *network.Network {
	hiddenNodes := 30
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with quadratic cost and sigmoid activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)
	net := network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(dims(d.NumInputs), hiddenNodes, network.Linear)
	net.AddLayer([]int{hiddenNodes}, d.NumOutputs, network.Sigmoid)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return net
}

type Loader2 struct{ Loader }

func (l Loader2) DefaultConfig() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  100,
		BatchSize: 250,
		LearnRate: 0.5,
		Momentum:  0.7,
		StopAfter: 8,
		LogEvery:  5,
		Sampler:   "random",
	}
}

func (l Loader2) CreateNetwork(cfg *network.Config, d *data.Dataset) *network.Network {
	hiddenNodes := 400
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with cross entropy cost and relu activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)
	net := network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(dims(d.NumInputs), hiddenNodes, network.Linear)
	net.AddLayer(dims(hiddenNodes), d.NumOutputs, network.Relu)
	net.AddCrossEntropyOutput(d.NumOutputs)
	return net
}

func dims(n int) []int {
	size := int(math.Sqrt(float64(n)))
	if size*size == n {
		return []int{size, size}
	}
	return []int{n}
}
