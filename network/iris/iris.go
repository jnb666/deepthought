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

func (l Loader) Dataset() string {
	return "iris"
}

func (l Loader) Load(d *data.Dataset) (cfg *network.Config, net *network.Network) {
	cfg = &network.Config{
		MaxEpoch:  200,
		LearnRate: 2.0,
		Threshold: 0.1,
		LogEvery:  25,
	}
	fmt.Println("IRIS DATASET: single layer with quadratic cost")
	net = network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer(d.NumInputs, d.NumOutputs, network.Linear)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return
}
