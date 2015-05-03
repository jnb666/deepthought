package main

import (
	"flag"
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
	"os"
	"runtime"
)

const (
	width  = 800
	height = 800
)

var (
	debug     = false
	batch     = false
	logEvery  = 0
	trainRuns = 20
	seed      = int64(1)
)

// exit if fatal error
func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// return function with stopping criteria
func stopCriteria(s *network.Stats) bool {
	var cost float64
	if s.Valid.Error.Len() > 0 {
		cost = s.Valid.Error.Last()
	} else {
		cost = s.Train.Error.Last()
	}
	done := s.Epoch >= maxEpoch || cost < threshold
	if logEvery > 0 && (s.Epoch%logEvery == 0 || done) {
		fmt.Println(s)
	}
	return done
}

// train the network
func train(net *network.Network, data data.Dataset, s *network.Stats) {
	failed := 0
	for run := 0; run < trainRuns; run++ {
		if !batch && run > 0 {
			fmt.Print("hit return for next run")
			fmt.Fscanln(os.Stdin)
		}
		net.SetRandomWeights()
		if debug {
			fmt.Println(net)
		}
		epoch := net.Train(data, float32(learnRate), s, stopCriteria)
		status := "**SUCCESS**"
		if epoch >= maxEpoch-1 {
			status = "**FAILED **"
			failed++
		}
		fmt.Printf("%s  epochs=%4d  run time=%.3f  reg error=%.3f  class error=%.3f\n",
			status, epoch, s.RunTime.Last(), s.RegError.Last(), s.ClsError.Last())
		if debug {
			fmt.Println(net)
			//fmt.Println(net.FeedForward(data.Train.Input))
		}
	}
	fmt.Printf("\n== success rate: %.0f%% ==\nnum epochs:  %s\nrun time:    %s\nreg error:   %s\nclass error: %s\n",
		100*float64(trainRuns-failed)/float64(trainRuns), s.NumEpochs, s.RunTime, s.RegError, s.ClsError)
}

// setup the plots
func createPlots(stats *network.Stats, d data.Dataset) (rows, cols int, plots []*mplot.Plot) {
	p1 := mplot.New()
	p1.Title.Text = fmt.Sprintf("Learning rate = %g", learnRate)
	pError, pClass := stats.ErrorPlots(d)
	p1.X.Label.Text = "epoch"
	p1.Y.Label.Text = "average error"
	mplot.AddLines(p1, pError...)
	p2 := mplot.New()
	p2.X.Label.Text = "epoch"
	p2.Y.Label.Text = "classification error"
	mplot.AddLines(p2, pClass...)
	p3 := mplot.New()
	p3.Title.Text = "Mean value over runs"
	p3.X.Label.Text = "run number"
	p3.Y.Label.Text = "num epochs"
	mplot.AddLines(p3, mplot.NewLine(stats.NumEpochs.Vector, ""))
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
	flag.BoolVar(&debug, "debug", debug, "enable debug printing and gradient checks")
	flag.BoolVar(&batch, "batch", batch, "set batch mode to disable plotting")
	flag.Int64Var(&seed, "seed", seed, "random number seed - or 0 to use current time")
	flag.IntVar(&maxEpoch, "epochs", maxEpoch, "maximum number of epochs")
	flag.IntVar(&logEvery, "log", logEvery, "log stats every n epochs or only at end of run if 0")
	flag.IntVar(&trainRuns, "runs", trainRuns, "no. of training runs")
	flag.Float64Var(&learnRate, "eta", learnRate, "network learning rate")
	flag.Parse()
	if !batch {
		runtime.LockOSThread()
	}

	// setup the network
	network.SeedRandom(seed)
	stats := network.NewStats(maxEpoch, trainRuns)
	data, net := setup()

	if batch {
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
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
