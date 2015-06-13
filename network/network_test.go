package network_test

import (
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"
	_ "github.com/jnb666/deepthought/network/iris"
	"testing"
)

func init() {
	network.Init(blas.Native32)
}

func TestNetwork(t *testing.T) {
	_, net, _, err := network.Load("iris", 10)
	if err != nil {
		t.Fatal(err)
	}
	net.SetRandomWeights()
	t.Log(net)
}

func TestFeedForward(t *testing.T) {
	_, net, d, err := network.Load("iris", 10)
	if err != nil {
		t.Fatal(err)
	}
	net.SetRandomWeights()
	t.Log(net)
	t.Logf("input:\n%s\n", d.Test.Input)
	output := net.FeedForward(d.Test.Input)
	t.Logf("output:\n%s\n", output)
	if output.Rows() != 10 || output.Cols() != 3 {
		t.Error("output is wrong size!")
	}
}

func TestTrain(t *testing.T) {
	cfg, net, d, err := network.Load("iris", 0)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Print()
	net.SetRandomWeights()
	t.Log(net)
	t.Logf("read %d test %d train and %d validation samples: max=%d\n",
		d.Test.NumSamples, d.Train.NumSamples, d.Valid.NumSamples, d.MaxSamples)
	stopFunc := network.StopCriteria(cfg)
	s := network.NewStats()
	s.StartRun()
	var done, failed bool
	for !done {
		net.Train(s, d, cfg)
		s.Update(net, d)
		done, failed = stopFunc(s)
	}
	t.Log(s.EndRun(failed))
	if failed {
		t.Error("training failed to hit threshold!")
	}
	t.Log(net)
}
