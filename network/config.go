package network

import (
	"fmt"
	"github.com/jnb666/deepthought/config"
	"github.com/jnb666/deepthought/data"
	"sort"
)

// Register map of all loaders which are available
var Register = map[string]Loader{}

type Config struct {
	MaxRuns     int     // number of runs: required
	MaxEpoch    int     // maximum epoch: required
	LearnRate   float64 // learning rate eta: required
	WeightDecay float64 // weight decay epsilon
	Momentum    float64 // momentum term used in weight updates
	Threshold   float64 // target cost threshold
	BatchSize   int     // minibatch size
	StopAfter   int     // stop after n epochs with no improvement
	LogEvery    int     // log stats every n epochs
	Sampler     string  // sampler to use
}

func (c *Config) Print() {
	config.Print(c)
}

// Data sets function lists all the registered models.
func DataSets() (s []string) {
	for name := range Register {
		s = append(s, name)
	}
	sort.Strings(s)
	return s
}

// Loader type is used to load new data sets and associated config.
type Loader interface {
	DefaultConfig() *Config
	DatasetName() string
	CreateNetwork(cfg *Config, d *data.Dataset) *Network
}

// Load function loads a data set, creates the network and returns the default config
func Load(name string) (cfg *Config, net *Network, d *data.Dataset, err error) {
	loader, ok := Register[name]
	if !ok {
		err = fmt.Errorf("Load: unknown dataset name %s\n", name)
		return
	}
	cfg = loader.DefaultConfig()
	config.Load(cfg, name)
	if d, err = data.Load(loader.DatasetName(), 0); err != nil {
		return
	}
	net = loader.CreateNetwork(cfg, d)
	return
}

// Stop criteria function returns a function to check if training is complete.
func StopCriteria(cfg *Config) func(*Stats) (done, failed bool) {
	prevCost := NewBuffer(cfg.StopAfter)
	return func(s *Stats) (done, failed bool) {
		var cost float64
		if s.Valid.Error.Len() > 0 {
			cost = s.Valid.Error.Last()
		} else {
			cost = s.Train.Error.Last()
		}
		if s.Epoch >= cfg.MaxEpoch {
			done = true
			failed = true
		} else if cost <= cfg.Threshold {
			done = true
		} else if cfg.StopAfter > 0 {
			if s.Epoch > cfg.StopAfter {
				done = cost > prevCost.Max()
			}
			prevCost.Push(cost)
		}
		return
	}
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
