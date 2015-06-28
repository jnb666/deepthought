package network

import (
	"github.com/jnb666/deepthought/blas"
	"math/rand"
)

var samplers = map[string]Sampler{
	"uniform": &uniformSampler{},
	"random":  &randomSampler{},
}

var SamplerNames = []string{"uniform", "random"}

// Sampler interface is used to split the data set into batches.
type Sampler interface {
	Init(samples, batchSize int) Sampler
	Next() bool
	Sample(in, out blas.Matrix)
	Release()
}

// NewSampler function creates a new sampler of the given type.
func NewSampler(typ string) Sampler {
	if s, ok := samplers[typ]; ok {
		return s
	}
	panic("sampler of type " + typ + " not found")
}

// uniformSampler loops over minibatches in order with no randomisation.
type uniformSampler struct {
	samples int
	batch   int
	start   int
}

func (s *uniformSampler) Init(samples, batchSize int) Sampler {
	s.samples = samples
	s.batch = batchSize
	s.start = 0
	return s
}

func (s *uniformSampler) Next() bool {
	if s.start+s.batch < s.samples {
		s.start += s.batch
		return true
	}
	return false
}

func (s *uniformSampler) Sample(in, out blas.Matrix) {
	end := s.start + s.batch
	if in.Rows() < end {
		end = in.Rows()
	}
	out.Copy(in.Row(s.start, end), nil)
}

func (s *uniformSampler) Release() {}

// randomSampler shuffles the indices randomly on reach run.
type randomSampler struct {
	index blas.Matrix
	batch int
	start int
}

func (s *randomSampler) Init(samples, batchSize int) Sampler {
	s.batch = batchSize
	s.start = 0
	s.index = blas.New(samples, 1)
	data := make([]float32, samples)
	for i, ix := range rand.Perm(samples) {
		data[i] = float32(ix)
	}
	s.index.Load(blas.RowMajor, data...)
	return s
}

func (s *randomSampler) Next() bool {
	if s.start+s.batch < s.index.Rows() {
		s.start += s.batch
		return true
	}
	return false
}

func (s *randomSampler) Sample(in, out blas.Matrix) {
	end := s.start + s.batch
	if s.index.Rows() < end {
		end = s.index.Rows()
	}
	out.Copy(in, s.index.Row(s.start, end))
}

func (s *randomSampler) Release() {
	s.index.Release()
}
