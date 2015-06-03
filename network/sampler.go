package network

import (
	"github.com/jnb666/deepthought/blas"
	"math/rand"
)

var Samplers = map[string]Sampler{
	"uniform": &UniformSampler{},
	"random":  &RandomSampler{},
}

var SamplerNames = []string{"uniform", "random"}

// Sampler interface is used to split the data set into batches.
type Sampler interface {
	Init(samples, batchSize int) Sampler
	Next() bool
	Sample(in, out blas.Matrix)
	Release()
}

// UniformSampler loops over minibatches in order with no randomisation.
type UniformSampler struct {
	samples int
	batch   int
	start   int
}

func (s *UniformSampler) Init(samples, batchSize int) Sampler {
	s.samples = samples
	s.batch = batchSize
	s.start = 0
	return s
}

func (s *UniformSampler) Next() bool {
	if s.start+s.batch < s.samples {
		s.start += s.batch
		return true
	}
	return false
}

func (s *UniformSampler) Sample(in, out blas.Matrix) {
	end := s.start + s.batch
	if in.Rows() < end {
		end = in.Rows()
	}
	out.Copy(in.Row(s.start, end), nil)
}

func (s *UniformSampler) Release() {}

// RandomSampler shuffles the indices randomly on reach run.
type RandomSampler struct {
	index blas.Matrix
	batch int
	start int
}

func (s *RandomSampler) Init(samples, batchSize int) Sampler {
	s.batch = batchSize
	s.start = 0
	s.index = blas.New(samples, 1)
	data := make([]float64, samples)
	for i, ix := range rand.Perm(samples) {
		data[i] = float64(ix)
	}
	s.index.Load(blas.RowMajor, data...)
	return s
}

func (s *RandomSampler) Next() bool {
	if s.start+s.batch < s.index.Rows() {
		s.start += s.batch
		return true
	}
	return false
}

func (s *RandomSampler) Sample(in, out blas.Matrix) {
	end := s.start + s.batch
	if s.index.Rows() < end {
		end = s.index.Rows()
	}
	out.Copy(in, s.index.Row(s.start, end))
}

func (s *RandomSampler) Release() {
	s.index.Release()
}
