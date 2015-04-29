package iris

import (
	"errors"
	"github.com/jnb666/deepthought/data"
)

// Data directory
var base = "/home/john/go/src/github.com/jnb666/deepthought/data/iris/"

// register dataset when module is imported
func init() {
	data.Register["iris"] = loader{}
}

type loader struct{}

// Load function loads and returns the iris dataset.
// samples is maxiumum number of records to load from each dataset if non-zero.
func (loader) Load(samples int) (s data.Dataset, err error) {
	var nin, nout int
	s.Test, s.NumInputs, s.NumOutputs, err = data.LoadFile(base+"iris_test.dat", samples)
	if err != nil {
		return
	}
	s.MaxSamples = s.Test.NumSamples

	s.Train, nin, nout, err = data.LoadFile(base+"iris_training.dat", samples)
	if err != nil {
		return
	}
	if nin != s.NumInputs || nout != s.NumOutputs {
		return s, errors.New("mismatch in number of inputs or outputs in training set")
	}
	if s.Train.NumSamples > s.MaxSamples {
		s.MaxSamples = s.Train.NumSamples
	}

	s.Valid, nin, nout, err = data.LoadFile(base+"iris_validation.dat", samples)
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
