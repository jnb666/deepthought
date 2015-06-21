// Package iris loads the iris dataset from file.
package iris

import (
	"errors"
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
)

// Data directory
const base = "/home/john/go/src/github.com/jnb666/deepthought/network/iris/"

// register dataset when module is imported
func init() {
	network.Register("iris", Loader{})
}

type Classify struct{}

func (Classify) Apply(out, class blas.Matrix) blas.Matrix {
	return class.MaxCol(out)
}

type Loader struct{}

// Config returns the default configuration
func (Loader) Config() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  200,
		LearnRate: 10.0,
		Threshold: 0.1,
		LogEvery:  5,
		Sampler:   "uniform",
	}
}

// CreateNetwork instantiates a new network with given config.
func (l Loader) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
	fmt.Println("IRIS DATASET: single layer with quadratic cost")
	net := network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer([]int{d.NumInputs}, d.NumOutputs, network.Linear)
	net.AddQuadraticOutput(d.NumOutputs, network.Sigmoid)
	return net
}

// Load function loads and returns the iris dataset.
func (Loader) Load(samples int) (s *network.Dataset, err error) {
	var nin, nout int
	s = new(network.Dataset)
	s.OutputToClass = Classify{}
	s.Test, s.NumInputs, s.NumOutputs, err = network.LoadFile(base+"iris_test.dat", samples, s.OutputToClass)
	if err != nil {
		return
	}
	s.MaxSamples = s.Test.NumSamples

	s.Train, nin, nout, err = network.LoadFile(base+"iris_training.dat", samples, s.OutputToClass)
	if err != nil {
		return
	}
	if nin != s.NumInputs || nout != s.NumOutputs {
		return s, errors.New("mismatch in number of inputs or outputs in training set")
	}
	if s.Train.NumSamples > s.MaxSamples {
		s.MaxSamples = s.Train.NumSamples
	}

	s.Valid, nin, nout, err = network.LoadFile(base+"iris_validation.dat", samples, s.OutputToClass)
	if err != nil {
		return
	}
	if nin != s.NumInputs || nout != s.NumOutputs {
		return s, errors.New("mismatch in number of inputs or outputs in validation set")
	}
	if s.Valid.NumSamples > s.MaxSamples {
		s.MaxSamples = s.Valid.NumSamples
	}
	return
}

func (Loader) DistortTypes() (t []network.Distortion) { return }

func (Loader) Distort(in, out blas.Matrix, mask int, severity float64) {}

func (Loader) Release() {}
