package xor

import (
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
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
	// extra column for bias nodes
	s.Train.Input = m32.New(4, 2).Load(m32.RowMajor, 0, 0, 0, 1, 1, 0, 1, 1)
	s.Train.Output = m32.New(4, 1).Load(m32.RowMajor, 0, 1, 1, 0)
	s.Train.Classes = s.Train.Output
	s.OutputToClass = func(out, class *m32.Matrix) {
		class.Apply(out,
			func(x float32) float32 {
				if x > 0.5 {
					return 1
				}
				return 0
			})
	}
	return
}
