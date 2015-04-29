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
	width     = 800
	height    = 800
	maxError  = 0.4
	maxClsErr = 0.8
)

var (
	batch     = false
	logEvery  = 0
	trainRuns = 1
	seed      = int64(0)
)

// exit if fatal error
func checkErr(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// return function with stopping criteria
func stopCriteria(stats *network.Stats) func(int) bool {
	return func(epoch int) bool {
		done := epoch >= maxEpoch || stats.Valid.Error.Last() < threshold
		if logEvery > 0 && ((epoch+1)%logEvery == 0 || done) {
			fmt.Println(stats)
		}
		return done
	}
}

// train the network
func train(net *network.Network, data data.Dataset, s *network.Stats) {
	for run := 0; run < trainRuns; run++ {
		net.SetRandomWeights(float32(maxWeight))
		epoch := net.Train(data, float32(learnRate), s, stopCriteria(s))
		status := "**SUCCESS**"
		if epoch >= maxEpoch-1 {
			status = "**FAILED **"
		}
		fmt.Printf("%s  epochs=%3d  run time=%.3f  reg error=%.3f  class error=%.3f\n",
			status, epoch, s.RunTime.Last(), s.RegError.Last(), s.ClsError.Last())
	}
	fmt.Printf("\nrun time:    %s\nreg error:   %s\nclass error: %s\n", s.RunTime, s.RegError, s.ClsError)
}

func main() {
	// get params
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
	stats := network.NewStats(maxEpoch, trainRuns, maxError, maxClsErr)
	data, net := setup()

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
		mplot.AddLines(plt1,
			mplot.NewLine(stats.Train.Error, "training"),
			mplot.NewLine(stats.Valid.Error, "validation"),
			mplot.NewLine(stats.Test.Error, "test set"),
		)
		plt2 := mplot.New()
		plt2.X.Label.Text = "epoch"
		plt2.Y.Label.Text = "classification error"
		mplot.AddLines(plt2,
			mplot.NewLine(stats.Train.ClassError, "training"),
			mplot.NewLine(stats.Valid.ClassError, "validation"),
			mplot.NewLine(stats.Test.ClassError, "test set"),
		)
		plt3 := mplot.New()
		plt3.Title.Text = "Mean value over runs"
		plt3.X.Label.Text = "run number"
		plt3.Y.Label.Text = "run time"
		mplot.AddLines(plt3,
			mplot.NewLine(stats.RunTime.Vector, ""),
		)
		plt4 := mplot.New()
		plt4.X.Label.Text = "run number"
		plt4.Y.Label.Text = "average test error"
		mplot.AddLines(plt4,
			mplot.NewLine(stats.RegError.Vector, "reg error"),
			mplot.NewLine(stats.ClsError.Vector, "class error"),
		)
		go train(net, data, stats)

		for !window.ShouldClose() {
			plt3.Y.Min, plt3.Y.Max = 0, stats.RunTime.Max
			plt4.Y.Min, plt4.Y.Max = 0, max(stats.RegError.Max, stats.ClsError.Max)
			window.Draw(2, 2, plt1, plt3, plt2, plt4)
		}
	}
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
