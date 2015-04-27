package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"

	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
)

const (
	width     = 500
	height    = 1000
	threshold = 0.1
	maxWeight = 0.5
	samples   = 0
	ymax      = 0.8
)

var (
	maxEpoch  = 500
	batch     = false
	logEvery  = 0
	learnRate = 0.5
	trainRuns = 1
	pause     = 0
	seed      = int64(0)
)

// exit if fatal error
func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// train the network
func train(net *network.Network, data data.Dataset, stats *network.Stats) {
	runTime := &mplot.RunningStat{}
	regError := &mplot.RunningStat{}
	classError := &mplot.RunningStat{}

	// loop over training runs
	for i := 0; i < trainRuns; i++ {
		net.SetRandomWeights(maxWeight)
		start := time.Now()
		epoch := net.Train(data, float32(learnRate), stats,
			func(epoch int) bool {
				done := epoch >= maxEpoch || stats.Valid.Error.Last() < threshold
				if logEvery > 0 && ((epoch+1)%logEvery == 0 || done) {
					fmt.Println(stats)
				}
				return done
			})
		var status string
		if epoch < maxEpoch-1 {
			status = "**SUCCESS**"
		} else {
			status = "**FAILED **"
		}
		// update stats
		runTime.Push(time.Since(start).Seconds())
		regError.Push(stats.Test.Error.Last())
		classError.Push(stats.Test.ClassError.Last())
		fmt.Printf("%s  epochs=%3d  run time=%.3f  reg error=%.3f  class error=%.3f\n",
			status, epoch, runTime.Last, regError.Last, classError.Last)
		time.Sleep(time.Millisecond * time.Duration(pause))
	}
	// summary stats
	fmt.Printf("\nrun time:    %s\nreg error:   %s\nclass error: %s\n", runTime, regError, classError)
}

func main() {
	// get params
	flag.BoolVar(&batch, "batch", batch, "set batch mode to disable plotting")
	flag.Int64Var(&seed, "seed", seed, "random number seed - or 0 to use current time")
	flag.IntVar(&maxEpoch, "epochs", maxEpoch, "maximum number of epochs")
	flag.IntVar(&logEvery, "log", logEvery, "log stats every n epochs or only at end of run if 0")
	flag.IntVar(&trainRuns, "runs", trainRuns, "no. of training runs")
	flag.IntVar(&pause, "pause", pause, "pause before stating next run in milliseconds")
	flag.Float64Var(&learnRate, "eta", learnRate, "network learning rate")
	flag.Parse()
	if !batch {
		runtime.LockOSThread()
	}

	// seed random number generator
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("random seed =", seed)
	rand.Seed(seed)

	// load data sets
	data, err := iris.Load(samples)
	checkErr(err)

	// set up network
	net := network.NewNetwork()
	net.Add(network.NewFCLayer(data.NumInputs, data.NumOutputs))
	stats := network.NewStats(maxEpoch)

	if batch {
		train(net, data, stats)
	} else {
		// open window
		window, err := mplot.NewWindow(width, height, "trainer")
		checkErr(err)

		// settup plotter
		plt1 := mplot.New()
		plt1.Title.Text = fmt.Sprintf("Learning rate = %g", learnRate)
		plt1.X.Label.Text = "epoch"
		plt1.Y.Label.Text = "average error"
		plt2 := mplot.New()
		plt2.X.Label.Text = "epoch"
		plt2.Y.Label.Text = "validation error"

		mplot.AddLines(plt1,
			mplot.NewLine(stats.Train.Error, "training", 0, ymax),
			mplot.NewLine(stats.Valid.Error, "validation", 0, ymax),
			mplot.NewLine(stats.Test.Error, "test set", 0, ymax),
		)
		mplot.AddLines(plt2,
			mplot.NewLine(stats.Train.ClassError, "training", 0, ymax),
			mplot.NewLine(stats.Valid.ClassError, "validation", 0, ymax),
			mplot.NewLine(stats.Test.ClassError, "test set", 0, ymax),
		)

		go train(net, data, stats)

		for !window.ShouldClose() {
			window.Draw(2, 1, plt1, plt2)
		}
	}
}
