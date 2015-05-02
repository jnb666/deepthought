package xor

import (
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
)

func rescale(m *m32.Matrix) {
	m.Apply(m, func(x float32) float32 {
		return -0.5 + x
	})
}

func out2class(out, class *m32.Matrix) {
	class.Apply(out, func(x float32) float32 {
		if x > 0 {
			return 1
		} else {
			return 0
		}
	})
}

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
	s.OutputToClass = out2class
	s.Train = new(data.Data)
	s.Train.NumSamples = 4
	s.Train.Input = m32.New(4, 2).Load(m32.RowMajor, 0, 0, 0, 1, 1, 0, 1, 1)
	s.Train.Output = m32.New(4, 1).Load(m32.RowMajor, 0, 1, 1, 0)
	s.Train.Classes = m32.New(4, 1).Load(m32.RowMajor, 0, 1, 1, 0)
	rescale(s.Train.Input)
	rescale(s.Train.Output)
	return
}
