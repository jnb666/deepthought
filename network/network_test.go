package network

import (
	"github.com/jnb666/deepthought/data/iris"
	"testing"
)

func TestLayer(t *testing.T) {
	s, err := iris.Load(10)
	if err != nil {
		t.Fatal(err)
	}
	l := NewFCLayer(s.NumInputs, s.NumOutputs)
	l.Weights().Random(-0.5, 0.5)
	t.Log(l)
	t.Logf("input:\n%s\n", s.Test.Input)
	output := l.FeedForward(s.Test.Input)
	t.Logf("output:\n%s\n", output)
}

func TestTrain(t *testing.T) {
	s, err := iris.Load(1)
	if err != nil {
		t.Fatal(err)
	}
	net := NewNetwork()
	net.Add(NewFCLayer(s.NumInputs, s.NumOutputs))
	net.SetRandomWeights(0.5)
	t.Log(net)
	maxEpoch := 10
	stats := NewStats(maxEpoch)
	net.Train(s, 0.1, stats, func(ep int) bool { return ep < maxEpoch })
	t.Logf("Error:\n%s\n", stats.Test.Error)
	t.Logf("ClassError:\n%s\n", stats.Test.ClassError)
}
