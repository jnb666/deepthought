package network

import (
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"testing"
)

const maxEpoch = 200

func createNetwork(batchSize int) (net *Network, s data.Dataset, err error) {
	if s, err = data.Load("iris", batchSize); err != nil {
		return
	}
	net = NewNetwork(s.MaxSamples)
	net.InputLayer(s.NumInputs, s.NumOutputs)
	net.QuadraticOutput(s.NumOutputs, SigmoidActivation)
	net.SetRandomWeights()
	return
}

func TestNetwork(t *testing.T) {
	net, _, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(net)
}

func TestFeedForward(t *testing.T) {
	net, d, err := createNetwork(10)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(net)
	t.Logf("input:\n%s\n", d.Test.Input)
	output := net.FeedForward(d.Test.Input)
	t.Logf("output:\n%s\n", output)
}

func TestTrain(t *testing.T) {
	net, d, err := createNetwork(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("read %d test %d train and %d validation samples: max=%d\n",
		d.Test.NumSamples, d.Train.NumSamples, d.Valid.NumSamples, d.MaxSamples)
	t.Log(net)
	stats := NewStats(maxEpoch, 1)
	epochs := net.Train(d, 10, stats, func(ep int) bool {
		done := ep >= maxEpoch || stats.Valid.Error.Last() < 0.1
		if ep%10 == 0 || done {
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
