package network

import (
	"github.com/jnb666/deepthought/blas"
	"math/rand"
)

// Sampler interface is used to split the data set into batches.
type Sampler interface {
	Init(batchSize int) Sampler
	Next() bool
	Sample(in, out blas.Matrix)
	Release()
}

// UniformSampler creates a sampler which loops over minibatches in order with no randomisation.
func UniformSampler(numSamples int) Sampler {
	return &uniformSampler{samples: numSamples}
}

type uniformSampler struct {
	samples int
	batch   int
	start   int
}

func (s *uniformSampler) Init(batchSize int) Sampler {
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

// RandomSampler creates a sampler which shuffles the indices randomly on reach run.
func RandomSampler(numSamples int) Sampler {
	return &randomSampler{index: blas.New(numSamples, 1)}
}

type randomSampler struct {
	index blas.Matrix
	batch int
	start int
}

func (s *randomSampler) Init(batchSize int) Sampler {
	s.batch = batchSize
	s.start = 0
	rows := s.index.Rows()
	data := make([]float64, rows)
	for i, ix := range rand.Perm(rows) {
		data[i] = float64(ix)
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

// Buffer type is a fixed size circular buffer.
type Buffer struct {
	data []float64
	size int
}

// NewBuffer function creates a new buffer with allocated maximum size.
func NewBuffer(size int) *Buffer {
	return &Buffer{data: make([]float64, size)}
}

// Push method appends an item to the buffer.
func (b *Buffer) Push(v float64) {
	if b.size < len(b.data) {
		b.data[b.size] = v
		b.size++
	} else {
		copy(b.data, b.data[1:])
		b.data[b.size-1] = v
	}
}

// Len method returns the number of items in the buffer.
func (b *Buffer) Len() int {
	return b.size
}

// Max method returns the maximum value.
func (b *Buffer) Max() float64 {
	max := -1.0e99
	for _, v := range b.data {
		if v > max {
			max = v
		}
	}
	return max
}
