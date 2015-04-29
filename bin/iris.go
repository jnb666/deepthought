package main

import (
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"github.com/jnb666/deepthought/network"
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
