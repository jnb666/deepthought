package iris

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/mnist"
	"github.com/jnb666/deepthought/network"
)

func init() {
	network.Register["mnist"] = Loader{}
	network.Register["mnist2"] = Loader2{}
}

type Loader struct{}

func (l Loader) Dataset() string {
	return "mnist"
}

func (l Loader) Load(d *data.Dataset) (cfg *network.Config, net *network.Network) {
	hiddenNodes := 30
	cfg = &network.Config{
		MaxEpoch:  50,
		BatchSize: 100,
		LearnRate: 3.0,
		Threshold: 0.0067,
		LogEvery:  1,
	}
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with quadratic cost and sigmoid activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)

	net = network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(d.NumInputs, hiddenNodes, network.Linear)
	net.AddLayer(hiddenNodes, d.NumOutputs, network.Sigmoid)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return
}

type Loader2 struct{ Loader }

func (l Loader2) Load(d *data.Dataset) (cfg *network.Config, net *network.Network) {
	hiddenNodes := 400
	cfg = &network.Config{
		MaxEpoch:  100,
		BatchSize: 250,
		LearnRate: 0.5,
		Momentum:  0.7,
		StopAfter: 8,
		LogEvery:  5,
	}
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with cross entropy cost and relu activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)

	net = network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(d.NumInputs, hiddenNodes, network.Linear)
	net.AddLayer(hiddenNodes, d.NumOutputs, network.Relu)
	net.AddCrossEntropyOutput(d.NumOutputs)
	return
}
