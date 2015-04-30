package main

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/xor"
	"github.com/jnb666/deepthought/mplot"
	"github.com/jnb666/deepthought/network"
)

const (
	width  = 400
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
	d, err := data.Load("xor", 0)
	checkErr(err)
	fmt.Println(d.Train)
	net := network.NewNetwork(d.MaxSamples)
	net.Add(network.NewFCLayer(d.NumInputs, 2, d.MaxSamples))
	net.Add(network.NewFCLayer(2, d.NumOutputs, d.MaxSamples))
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
	return 2, 1, []*mplot.Plot{p1, p2}
}
