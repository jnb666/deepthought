// Package mnist loads the MNist dataset of handwritten digits.
package mnist

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	"math"
)

const (
	// Data directory
	base        = "/home/john/go/src/github.com/jnb666/deepthought/network/mnist/"
	trainImages = "train-images-idx3-ubyte"
	trainLabels = "train-labels-idx1-ubyte"
	testImages  = "t10k-images-idx3-ubyte"
	testLabels  = "t10k-labels-idx1-ubyte"
	numOutputs  = 10
	trainMax    = 50000
	testMax     = 10000
)

// register dataset when module is imported
func init() {
	network.Register("mnist", &Loader{})
	network.Register("mnist2", Loader2{&Loader{}})
}

// classification function
type Classify struct{}

func (Classify) Apply(out, class blas.Matrix) blas.Matrix { return class.MaxCol(out) }

// default configuration
func (*Loader) Config() *network.Config {
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

func (*Loader) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
	hiddenNodes := 30
	fmt.Printf("MNIST DATASET: [%d,%d,%d] layers with quadratic cost and sigmoid activation\n",
		d.NumInputs, hiddenNodes, d.NumOutputs)
	net := network.New(cfg.BatchSize, d.OutputToClass)
	net.AddLayer(dims(d.NumInputs), hiddenNodes, network.Linear)
	net.AddLayer([]int{hiddenNodes}, d.NumOutputs, network.Sigmoid)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return net
}

// Alternative configuration
type Loader2 struct{ *Loader }

func (Loader2) Config() *network.Config {
	return &network.Config{
		MaxRuns:    1,
		MaxEpoch:   100,
		BatchSize:  250,
		LearnRate:  0.5,
		Momentum:   0.7,
		StopAfter:  8,
		LogEvery:   5,
		Sampler:    "random",
		Distortion: 1,
	}
}

func (Loader2) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
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
