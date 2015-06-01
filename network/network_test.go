package network

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	_ "github.com/jnb666/deepthought/data/iris"
	"testing"
)

func init() {
	Init(blas.Native32)
	//Init(blas.OpenCL32)
}

func createNetwork(samples int) (net *Network, s *data.Dataset, err error) {
	if s, err = data.Load("iris", samples); err != nil {
		return
	}
	net = New(s.MaxSamples, s.OutputToClass)
	net.AddLayer(s.NumInputs, s.NumOutputs, Linear)
	net.AddQuadraticOutput(s.NumOutputs, Sigmoid)
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
	if output.Rows() != 10 || output.Cols() != 3 {
		t.Error("output is wrong size!")
	}
}

func TestTrain(t *testing.T) {
	net, d, err := createNetwork(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("read %d test %d train and %d validation samples: max=%d\n",
		d.Test.NumSamples, d.Train.NumSamples, d.Valid.NumSamples, d.MaxSamples)
	t.Log(net)
	cfg := &Config{
		MaxEpoch:  200,
		LearnRate: 10,
		Threshold: 0.1,
		LogEvery:  5,
		Sampler:   UniformSampler(d.Train.NumSamples),
	}
	stopFunc := StopCriteria(cfg)
	s := NewStats()
	s.StartRun()
	var done, failed bool
	for !done {
		net.Train(s, d, cfg)
		s.Update(net, d)
		done, failed = stopFunc(s)
		if s.Epoch%cfg.LogEvery == 0 || done {
			t.Log(s)
		}
	}
	t.Log(s.EndRun(failed))
	if failed {
		t.Error("training failed to hit threshold!")
	}
	t.Log(net)
}
