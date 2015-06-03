package main

import (
	"flag"
	"fmt"

	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/network"
	"github.com/jnb666/deepthought/qml"

	_ "github.com/jnb666/deepthought/network/iris"
	_ "github.com/jnb666/deepthought/network/mnist"
	_ "github.com/jnb666/deepthought/network/xor"
)

func main() {
	var seed int64
	network.Init(blas.OpenCL32)
	dataSets := network.DataSets()
	model := dataSets[0]
	flag.StringVar(&model, "model", model, "data model to run")
	flag.Int64Var(&seed, "seed", 0, "random number seed")
	flag.Parse()

	cfg, net, data, err := network.Load(model)
	if err != nil {
		fmt.Println(err)
		return
	}
	network.SeedRandom(seed)
	cfg.Print()
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

	ctrl := qml.NewCtrl(cfg, dataSets, model)
	go train(cfg, net, data, s, ctrl)
	qml.MainLoop(ctrl, p1, p2, p3, p4)
	ctrl.WG.Wait()
}

// train the network
func train(cfg *network.Config, net *network.Network, data *data.Dataset, s *network.Stats, ctrl *qml.Ctrl) {
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
			net.Train(s, data, cfg)
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
			cfg, net, data, _ = network.Load(ev.Arg)
			cfg.Print()
			ctrl.Refresh(cfg)
		case "quit": // exit the program
			net.Release()
			blas.Release()
			ctrl.WG.Done()
			return
		}
	}
}
