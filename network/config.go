package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"reflect"
	"sort"
	"strings"
)

type Config struct {
	MaxEpoch    int     // maximum epoch: required
	LearnRate   float64 // learning rate eta: required
	WeightDecay float64 // weight decay epsilon
	Momentum    float64 // momentum term used in weight updates
	Threshold   float64 // target cost threshold
	BatchSize   int     // minibatch size
	StopAfter   int     // stop after n epochs with no improvement
	LogEvery    int     // log stats every n epochs
	RandomSeed  int64   // random number seed - set randomly if zero
	BatchMode   bool    // turns off plotting
	Debug       bool    // enable debug printing
	Sampler     Sampler // sampler to use
}

// Register map of all loaders which are available
var Register = map[string]Loader{}

func DataSets() (s []string) {
	for name := range Register {
		s = append(s, name)
	}
	sort.Strings(s)
	return s
}

type Loader interface {
	Load(d *data.Dataset) (cfg *Config, net *Network)
	Dataset() string
}

// Load function loads a data set, creates the network and returns the default config
func Load(name string) (cfg *Config, net *Network, d *data.Dataset, err error) {
	loader, ok := Register[name]
	if !ok {
		err = fmt.Errorf("Load: unknown dataset name %s\n", name)
		return
	}
	if d, err = data.Load(loader.Dataset(), 0); err != nil {
		return
	}
	cfg, net = loader.Load(d)
	cfg.RandomSeed = SeedRandom(cfg.RandomSeed)
	if cfg.BatchSize > 0 && d.Train.NumSamples > cfg.BatchSize {
		cfg.Sampler = RandomSampler(d.Train.NumSamples)
	} else {
		cfg.Sampler = UniformSampler(d.Train.NumSamples)
	}
	//if cfg.Debug {
	//	net.CheckGradient(cfg.LogEvery, 1e-6, 0, 1)
	//}
	fmt.Println(cfg, "\n")
	return
}

// String method formats the config for printing.
func (c *Config) String() string {
	s := reflect.ValueOf(c).Elem()
	str := make([]string, s.NumField())
	for i := 0; i < s.NumField(); i++ {
		str[i] = fmt.Sprintf("%12s : %v", s.Type().Field(i).Name, s.Field(i).Interface())
	}
	return strings.Join(str, "\n")
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
