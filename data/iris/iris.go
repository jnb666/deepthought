package iris

import (
	"github.com/jnb666/deepthought/data"
)

// Registered datasets
var base = "/home/john/go/src/github.com/jnb666/deepthought/data/iris/"

// Load function loads and returns the iris dataset.
// samples is maxiumum number of records to load from each dataset if non-zero.
func Load(samples int) (s data.Dataset, err error) {
	if s.Test, s.NumInputs, s.NumOutputs, err = data.Load(base+"iris_test.dat", samples); err != nil {
		return
	}
	if s.Train, _, _, err = data.Load(base+"iris_training.dat", samples); err != nil {
		return
	}
	s.Valid, _, _, err = data.Load(base+"iris_validation.dat", samples)
	return
}
