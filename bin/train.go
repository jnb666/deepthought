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
	stats := network.NewStats(maxEpoch, trainRuns)
	data, net := setup()

	if batch {
		train(net, data, stats)
	} else {
		window, err := mplot.NewWindow(width, height, "trainer")
		checkErr(err)
		rows, cols, plt := createPlots(stats)

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
