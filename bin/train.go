package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
)

const (
	width  = 800
	height = 800
)

type Config struct {
	TrainRuns   int     // number of training runs: required
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
}

func (c Config) String() string {
	s := reflect.ValueOf(&c).Elem()
	str := make([]string, s.NumField())
	for i := 0; i < s.NumField(); i++ {
		str[i] = fmt.Sprintf("%12s : %v", s.Type().Field(i).Name, s.Field(i).Interface())
	}
	return strings.Join(str, "\n")
}

// exit if fatal error
func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// returns function with stopping criteria
func stopCriteria() func(*network.Stats) bool {
	prevCost := network.NewBuffer(cfg.StopAfter)
	return func(s *network.Stats) bool {
		var cost float64
		if s.Valid.Error.Len() > 0 {
			cost = s.Valid.Error.Last()
		} else {
			cost = s.Train.Error.Last()
		}
		var done bool
		if s.Epoch >= cfg.MaxEpoch || cost <= cfg.Threshold {
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
		return done
	}
}

// train the network
func train(net *network.Network, data *data.Dataset, s *network.Stats) {
	failed := 0
	sampler := network.UniformSampler(data.Train.NumSamples)
	if cfg.BatchSize > 0 && data.Train.NumSamples > cfg.BatchSize {
		sampler = network.RandomSampler(data.Train.NumSamples)
	}
	for run := 0; run < cfg.TrainRuns; run++ {
		if !cfg.BatchMode && run > 0 {
			time.Sleep(5 * time.Second)
		}
		net.SetRandomWeights()
		if cfg.Debug {
			fmt.Println(net)
		}
		epoch := net.Train(data, sampler, cfg.LearnRate, cfg.WeightDecay, cfg.Momentum, s, stopCriteria())
		status := "**SUCCESS**"
		if epoch >= cfg.MaxEpoch-1 {
			status = "**FAILED **"
			failed++
		}
		fmt.Printf("%s  epochs=%4d  run time=%.2fs  reg error=%.4f  class error=%.1f%%\n",
			status, epoch, s.RunTime.Last(), s.RegError.Last(), 100*s.ClsError.Last())
		if cfg.Debug {
			fmt.Println(net)
		}
	}
	fmt.Printf("\n== success rate: %.0f%% ==\nnum epochs:  %s\nrun time:    %s\nreg error:   %s\nclass error: %s\n",
		100*float64(cfg.TrainRuns-failed)/float64(cfg.TrainRuns), s.NumEpochs, s.RunTime, s.RegError, s.ClsError)
	sampler.Release()
}

// setup the plots
func createPlots(stats *network.Stats, d *data.Dataset) (rows, cols int, plots []*mplot.Plot) {
	p1 := mplot.New()
	p1.Title.Text = fmt.Sprintf("Learning rate = %g", cfg.LearnRate)
	pError, pClass := stats.ErrorPlots(d)
	p1.X.Label.Text = "epoch"
	p1.Y.Label.Text = "cost function"
	mplot.AddLines(p1, pError...)
	p2 := mplot.New()
	p2.X.Label.Text = "epoch"
	p2.Y.Label.Text = "classification error"
	mplot.AddLines(p2, pClass...)
	p3 := mplot.New()
	p3.Title.Text = "Mean value over runs"
	p3.X.Label.Text = "run number"
	p3.Y.Label.Text = "run time (s)"
	mplot.AddLines(p3, mplot.NewLine(stats.RunTime.Vector, ""))
	p4 := mplot.New()
	p4.X.Label.Text = "run number"
	p4.Y.Label.Text = "average test error"
	mplot.AddLines(p4,
		mplot.NewLine(stats.RegError.Vector, "reg error"),
		mplot.NewLine(stats.ClsError.Vector, "class error"),
	)
	return 2, 2, []*mplot.Plot{p1, p3, p2, p4}
}

func main() {
	// get params
	flag.BoolVar(&cfg.Debug, "debug", cfg.Debug, "enable debug printing and gradient checks")
	flag.BoolVar(&cfg.BatchMode, "batch", cfg.BatchMode, "set batch mode to disable plotting")
	flag.Int64Var(&cfg.RandomSeed, "seed", cfg.RandomSeed, "random number seed - or 0 to use current time")
	flag.IntVar(&cfg.MaxEpoch, "epochs", cfg.MaxEpoch, "maximum number of epochs")
	flag.IntVar(&cfg.LogEvery, "log", cfg.LogEvery, "log stats every n epochs or only at end of run if 0")
	flag.IntVar(&cfg.TrainRuns, "runs", cfg.TrainRuns, "no. of training runs")
	flag.Float64Var(&cfg.LearnRate, "eta", cfg.LearnRate, "network learning rate")
	flag.Parse()
	if !cfg.BatchMode {
		runtime.LockOSThread()
	}
	cfg.RandomSeed = network.SeedRandom(cfg.RandomSeed)

	// setup the network
	data, net := setup()
	stats := network.NewStats(cfg.MaxEpoch, cfg.TrainRuns)

	// run cleanup handler on exit
	c := make(chan os.Signal, 10)
	signal.Notify(c, os.Interrupt)
	go func() {
		<-c
		net.Release()
		blas.Release()
		os.Exit(1)
	}()

	if cfg.BatchMode {
		train(net, data, stats)
	} else {
		window, err := mplot.NewWindow(width, height, "trainer")
		checkErr(err)
		rows, cols, plt := createPlots(stats, data)
		go train(net, data, stats)
		for !window.ShouldClose() {
			window.Draw(rows, cols, plt...)
		}
	}
	net.Release()
	blas.Release()
}
