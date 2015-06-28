package main

import (
	"flag"
	"fmt"

	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/network"

	_ "github.com/jnb666/deepthought/network/iris"
	_ "github.com/jnb666/deepthought/network/mnist"
	_ "github.com/jnb666/deepthought/network/xor"
)

func main() {
	var debug bool
	var runs, maxEpoch int
	var seed int64
	network.Init(blas.OpenCL32)
	dataSets := network.DataSets()
	model := dataSets[0]
	flag.StringVar(&model, "model", model, "data model to run")
	flag.IntVar(&runs, "runs", 0, "number of runs")
	flag.IntVar(&maxEpoch, "epochs", 0, "maximum number of epochs")
	flag.Int64Var(&seed, "seed", 0, "random number seed")
	flag.BoolVar(&debug, "debug", false, "enable debug output")
	flag.Parse()
	cfg, net, data, err := network.Load(model, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	if runs > 0 {
		cfg.MaxRuns = runs
	}
	if maxEpoch > 0 {
		cfg.MaxEpoch = maxEpoch
	}
	//net.Verbose = true
	seed = blas.SeedRandom(seed)
	fmt.Println("set random seed to", seed)
	cfg.Print()
	s := network.NewStats()
	if debug {
		net.CheckGradient(5, 1e-4, 0, 5)
	}
	for i := 0; i < cfg.MaxRuns; i++ {
		net.SetRandomWeights()
		if debug {
			fmt.Println(net)
		}
		stop := network.StopCriteria(cfg)
		s.StartRun()
		var done, failed bool
		for !done {
			net.Train(s, data, cfg)
			s.Update(net, data)
			done, failed = stop(s)
		}
		if debug {
			fmt.Println(net)
		}
		fmt.Println(s.EndRun(failed))
	}
	fmt.Println(s.History())
	net.Release()
	data.Release()
	blas.Release()
}
