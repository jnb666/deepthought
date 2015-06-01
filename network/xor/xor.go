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

func (l Loader) Dataset() string {
	return "xor"
}

func (l Loader) Load(d *data.Dataset) (cfg *network.Config, net *network.Network) {
	cfg = &network.Config{
		MaxEpoch:  1000,
		LearnRate: 0.5,
		Threshold: 0.05,
		LogEvery:  25,
	}
	fmt.Printf("XOR DATASET: [2,2,1] layers with quadratic cost and tanh activation\n%s\n", cfg)

	net = network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer(2, 2, network.Linear)
	net.AddLayer(2, 1, network.Tanh)
	net.AddQuadraticOutput(1, network.Tanh)
	return
}
