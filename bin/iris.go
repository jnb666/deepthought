package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
)

const (
	width  = 800
	height = 800
)

// default params
var (
	threshold = 0.1
	maxWeight = 0.5
	maxEpoch  = 200
	learnRate = 0.5
)

// load the data and setup the network
func setup() (data.Dataset, *network.Network) {
	d, err := data.Load("iris", 0)
	checkErr(err)

	net := network.NewNetwork(d.MaxSamples)
	net.Add(network.NewFCLayer(d.NumInputs, d.NumOutputs, d.MaxSamples))
	return d, net
}

// setup the plots
func createPlots(stats *network.Stats) (rows, cols int, plots []*mplot.Plot) {
	p1 := mplot.New()
	p1.Title.Text = fmt.Sprintf("Learning rate = %g", learnRate)
	p1.X.Label.Text = "epoch"
	p1.Y.Label.Text = "average error"
	mplot.AddLines(p1,
		mplot.NewLine(stats.Train.Error, "training"),
		mplot.NewLine(stats.Valid.Error, "validation"),
		mplot.NewLine(stats.Test.Error, "test set"),
	)
	p2 := mplot.New()
	p2.X.Label.Text = "epoch"
	p2.Y.Label.Text = "classification error"
	mplot.AddLines(p2,
		mplot.NewLine(stats.Train.ClassError, "training"),
		mplot.NewLine(stats.Valid.ClassError, "validation"),
		mplot.NewLine(stats.Test.ClassError, "test set"),
	)
	p3 := mplot.New()
	p3.Title.Text = "Mean value over runs"
	p3.X.Label.Text = "run number"
	p3.Y.Label.Text = "run time"
	mplot.AddLines(p3,
		mplot.NewLine(stats.RunTime.Vector, ""),
	)
	p4 := mplot.New()
	p4.X.Label.Text = "run number"
	p4.Y.Label.Text = "average test error"
	mplot.AddLines(p4,
		mplot.NewLine(stats.RegError.Vector, "reg error"),
		mplot.NewLine(stats.ClsError.Vector, "class error"),
	)
	return 2, 2, []*mplot.Plot{p1, p3, p2, p4}
}
