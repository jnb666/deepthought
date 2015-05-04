package xor

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
)

// register dataset when module is imported
func init() {
	data.Register["xor"] = loader{}
}

type loader struct{}

// Load function loads and returns the xor dataset.
// samples is ignored, aleays returns 4 records
func (loader) Load(samples int) (s data.Dataset, err error) {
	s.MaxSamples = 4
	s.NumInputs = 2
	s.NumOutputs = 1
	s.Train = new(data.Data)
	s.Train.NumSamples = 4
	s.Train.Input = blas.New(4, 2).Load(blas.RowMajor, 0, 0, 0, 1, 1, 0, 1, 1)
	s.Train.Output = blas.New(4, 1).Load(blas.RowMajor, 0, 1, 1, 0)

	scale := blas.Unary64(func(x float64) float64 { return -0.5 + x })
	scale.Apply(s.Train.Input, s.Train.Input)
	scale.Apply(s.Train.Output, s.Train.Output)

	s.OutputToClass = blas.Unary64(func(x float64) float64 {
		if x > 0 {
			return 1
		} else {
			return 0
		}
	})
	s.Train.Classes = blas.New(4, 1)
	s.OutputToClass.Apply(s.Train.Output, s.Train.Classes)
	return
}
