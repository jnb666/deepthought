package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/network"
	"github.com/jnb666/deepthought/qml"

	_ "github.com/jnb666/deepthought/network/iris"
	_ "github.com/jnb666/deepthought/network/mnist"
	_ "github.com/jnb666/deepthought/network/xor"
)

func main() {
	network.Init(blas.OpenCL32)
	dataSets := network.DataSets()
	ix := 0
	flag.Parse()
	if flag.NArg() >= 1 {
		for i, name := range dataSets {
			if strings.ToLower(flag.Arg(0)) == strings.ToLower(name) {
				ix = i
			}
		}
	}
	cfg, net, data, sampler := loadNetwork(dataSets[ix])
	s := network.NewStats()

	p1 := qml.NewPlot("cost", "average cost vs epoch",
		qml.NewLine(s.Train.Error, "training"),
		qml.NewLine(s.Valid.Error, "validation"),
		qml.NewLine(s.Test.Error, "test set"),
	)
	p2 := qml.NewPlot("accuracy", "classification error vs epoch",
		qml.NewLine(s.Train.ClassError, "training"),
		qml.NewLine(s.Valid.ClassError, "validation"),
		qml.NewLine(s.Test.ClassError, "test set"),
	)
	p3 := qml.NewPlot("error hist", "histogram of cost per sample",
		qml.NewHistogram(s.Train.ErrorHist, "training"),
		qml.NewHistogram(s.Valid.ErrorHist, "validation"),
		qml.NewHistogram(s.Test.ErrorHist, "test set"),
	)
	p4 := qml.NewPlot("statistics", "performance vs run time in seconds",
		qml.NewPoints(s.RunTime, s.RegError, "average cost"),
		qml.NewPoints(s.RunTime, s.ClsError, "classification error"),
	)

	ctrl := qml.NewCtrl(dataSets, ix)
	go train(cfg, net, data, sampler, s, ctrl)
	qml.MainLoop(ctrl, p1, p2, p3, p4)
	ctrl.WG.Wait()
}

// initialise a new run
func loadNetwork(name string) (cfg *network.Config, net *network.Network, data *data.Dataset, smp network.Sampler) {
	var err error
	fmt.Println(">>load network", name)
	if cfg, net, data, err = network.Load(name); err != nil {
		panic(err)
	}
	smp = network.UniformSampler(data.Train.NumSamples)
	if cfg.BatchSize > 0 && data.Train.NumSamples > cfg.BatchSize {
		smp = network.RandomSampler(data.Train.NumSamples)
		fmt.Println("select random samper with batch size =", cfg.BatchSize)
	}
	return
}

// train the network
func train(cfg *network.Config, net *network.Network, data *data.Dataset, smp network.Sampler,
	s *network.Stats, ctrl *qml.Ctrl) {
	var running, started bool
	var stopCond func(*network.Stats) (bool, bool)

	startRun := func() {
		// start new training run
		net.SetRandomWeights()
		stopCond = network.StopCriteria(cfg)
		s.StartRun()
		started = true
	}

	endRun := func(failed bool) {
		// print stats at end of run
		started = false
		running = false
		ctrl.Done()
		fmt.Printf("%s\n\n", s.EndRun(failed))
	}

	for {
		ev := ctrl.NextEvent(running)
		switch ev.Typ {
		case "start": // start new run
			if started {
				endRun(true)
			}
			startRun()
		case "step": // step to next epoch
			if !started {
				startRun()
			}
			net.Train(s, data, smp, cfg)
			s.Update(net, data)
			done, failed := stopCond(s)
			if s.Epoch%cfg.LogEvery == 0 || done {
				fmt.Println(s)
			}
			if done {
				endRun(failed)
			}
		case "run": // toggle running mode
			running = ev.Arg == "start"
		case "stats": // print out the stats over runs
			fmt.Printf("%s\n\n", s.History())
		case "select": // choose a new data set
			s.Reset()
			cfg, net, data, smp = loadNetwork(ev.Arg)
			ctrl.Refresh()
		case "quit": // exit the program
			net.Release()
			blas.Release()
			ctrl.WG.Done()
			return
		}
	}
}
