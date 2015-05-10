package xor

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
)

const (
	c0 = -0.5
	c1 = 0.5
)

// register dataset when module is imported
func init() {
	data.Register["xor"] = loader{}
}

type loader struct{}

// Load function loads and returns the xor dataset.
// samples is ignored, aleays returns 4 records
func (loader) Load(samples, batch int) (s *data.Dataset, err error) {
	s = new(data.Dataset)
	s.MaxSamples = 4
	s.NumInputs = 2
	s.NumOutputs = 1
	s.Train = new(data.Data)
	s.Train.NumSamples = 4
	s.Train.Input = []blas.Matrix{blas.New(4, 2).Load(blas.RowMajor, c0, c0, c0, c1, c1, c0, c1, c1)}
	s.Train.Output = []blas.Matrix{blas.New(4, 1).Load(blas.RowMajor, c0, c1, c1, c0)}
	s.Train.Classes = []blas.Matrix{blas.New(4, 1).Load(blas.RowMajor, 0, 1, 1, 0)}
	if blas.Implementation() == blas.OpenCL32 {
		s.OutputToClass = blas.NewUnaryCL("x > 0.f")
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
