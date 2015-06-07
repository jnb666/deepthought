package network

import (
	"fmt"
	"github.com/jnb666/deepthought/config"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/vec"
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
	prevCost := vec.NewBuffer(cfg.StopAfter)
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
		if cfg.LogEvery > 0 && (s.Epoch%cfg.LogEvery == 0 || done) {
			fmt.Println(s)
		}
		return
	}
}
