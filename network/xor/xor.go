// Package xor loads the dataset for an xor gate.
package xor

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
)

const (
	c0 = -0.5
	c1 = 0.5
)

func init() {
	network.Register("xor", Loader{})
}

type Loader struct{}

// Config returns the default configuration
func (Loader) Config() *network.Config {
	return &network.Config{
		MaxRuns:   1,
		MaxEpoch:  1000,
		LearnRate: 0.5,
		Threshold: 0.05,
		LogEvery:  25,
		Sampler:   "uniform",
	}
}

// CreateNetwork instantiates a new network with given config.
func (Loader) CreateNetwork(cfg *network.Config, d *network.Dataset) *network.Network {
	fmt.Println("XOR DATASET: [2,2,1] layers with quadratic cost and tanh activation")
	net := network.New(d.MaxSamples, d.OutputToClass)
	net.AddLayer([]int{2}, 2, network.Linear)
	net.AddLayer([]int{2}, 1, network.Tanh)
	net.AddQuadraticOutput(1, network.Tanh)
	return net
}

// Load function loads and returns the xor dataset.
func (Loader) Load(samples int) (s *network.Dataset, err error) {
	s = new(network.Dataset)
	s.MaxSamples = 4
	s.NumInputs = 2
	s.NumOutputs = 1
	s.Train = new(network.Data)
	s.Train.NumSamples = 4
	s.Train.Input = blas.New(4, 2).Load(blas.RowMajor, c0, c0, c0, c1, c1, c0, c1, c1)
	s.Train.Output = blas.New(4, 1).Load(blas.RowMajor, c0, c1, c1, c0)
	s.Train.Classes = blas.New(4, 1).Load(blas.RowMajor, 0, 1, 1, 0)
	if blas.Implementation() == blas.OpenCL32 {
		s.OutputToClass = blas.NewUnaryCL("float y = x > 0.f;")
	} else {
		s.OutputToClass = blas.Unary64(func(x float64) float64 {
			if x > 0 {
				return 1
			} else {
				return 0
			}
		})
	}
	return
}

func (Loader) DistortTypes() (t []network.Distortion) { return }

func (Loader) Distort(in, out blas.Matrix, mask int, severity float64) {}

func (Loader) Release() {}
