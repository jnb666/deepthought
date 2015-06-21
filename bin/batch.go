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
	var runs int
	var seed int64
	network.Init(blas.OpenCL32)
	dataSets := network.DataSets()
	model := dataSets[0]
	flag.StringVar(&model, "model", model, "data model to run")
	flag.IntVar(&runs, "runs", 0, "number of runs")
	flag.Int64Var(&seed, "seed", 0, "random number seed")
	flag.Parse()

	cfg, net, data, err := network.Load(model, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	if runs > 0 {
		cfg.MaxRuns = runs
	}
	network.SeedRandom(seed)
	cfg.Print()
	s := network.NewStats()

	for i := 0; i < cfg.MaxRuns; i++ {
		net.SetRandomWeights()
		stop := network.StopCriteria(cfg)
		s.StartRun()
		var done, failed bool
		for !done {
			net.Train(s, data, cfg)
			s.Update(net, data)
			done, failed = stop(s)
		}
		fmt.Println(s.EndRun(failed))
	}
	fmt.Println(s.History())
	net.Release()
	data.Release()
	blas.Release()
}