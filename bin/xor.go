package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/xor"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
)

const (
	width  = 800
	height = 800
)

// default params
var (
	threshold = 0.02
	maxWeight = 1.0
	maxEpoch  = 1000
	learnRate = 0.1
)

// return function with stopping criteria
func stopCriteria(net *network.Network, stats *network.Stats) func(int) bool {
	return func(epoch int) bool {
		done := epoch >= maxEpoch || stats.Train.Error.Last() < threshold
		if logEvery > 0 && ((epoch+1)%logEvery == 0 || done) {
			fmt.Println(stats)
		}
		//if done {
		//	fmt.Println(net)
		//}
		return done
	}
}

// load the data and setup the network with one hidden layer
func setup() (data.Dataset, *network.Network) {
	d, err := data.Load("xor", 0)
	checkErr(err)
	fmt.Println(d.Train)
	net := network.NewNetwork(d.MaxSamples)
	net.InputLayer(2, 2)
	net.HiddenLayer(2, 1, network.SigmoidActivation)
	net.OutputLayer(1, network.SigmoidActivation)
	fmt.Println(net)
	return d, net
}

// setup the plots
func createPlots(stats *network.Stats) (rows, cols int, plots []*mplot.Plot) {
	p1 := mplot.New()
	p1.Title.Text = fmt.Sprintf("Learning rate = %g", learnRate)
	p1.X.Label.Text = "epoch"
	p1.Y.Label.Text = "average error"
	mplot.AddLines(p1, mplot.NewLine(stats.Train.Error, ""))
	p2 := mplot.New()
	p2.X.Label.Text = "epoch"
	p2.Y.Label.Text = "classification error"
	mplot.AddLines(p2, mplot.NewLine(stats.Train.ClassError, ""))
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
