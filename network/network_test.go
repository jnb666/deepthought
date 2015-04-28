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
	l := NewFCLayer(s.NumInputs, s.NumOutputs, 10)
	l.Weights().Random(-0.5, 0.5)
	t.Log(l)
	t.Logf("input:\n%s\n", s.Test.Input)
	output := l.FeedForward(s.Test.Input)
	t.Logf("output:\n%s\n", output)
}

func TestTrain(t *testing.T) {
	s, err := iris.Load(0)
	if err != nil {
		t.Fatal(err)
	}
	net := NewNetwork()
	net.Add(NewFCLayer(s.NumInputs, s.NumOutputs, s.MaxSamples))
	net.SetRandomWeights(0.5)
	t.Log(net)
	maxEpoch := 200
	stats := NewStats(maxEpoch)
	epochs := net.Train(s, 0.5, stats, func(ep int) bool {
		done := ep >= maxEpoch || stats.Valid.Error.Last() < 0.1
		if ep%25 == 0 || done {
			t.Log(stats)
		}
		return done
	})
	t.Logf("completed after %d epochs\n", epochs)
	if epochs == maxEpoch {
		t.Error("training failed to hit threshold!")
	}
	t.Log(net)
}
