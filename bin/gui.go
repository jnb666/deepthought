package main

import (
	"flag"
	"fmt"

	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/config"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/network"
	"github.com/jnb666/deepthought/qml"
	"github.com/jnb666/deepthought/vec"

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
	plts := createPlots(s)
	ctrl := qml.NewCtrl(cfg, dataSets, model, plts)
	go train(cfg, net, data, s, ctrl, &statsPlot{Plot: plts[3]})
	qml.MainLoop(ctrl)
	ctrl.WG.Wait()
}

// create the plots
func createPlots(s *network.Stats) []*qml.Plot {
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
	p4 := qml.NewPlot("statistics", "classification error vs run time in seconds")
	p4.Legend = qml.TopLeft
	return []*qml.Plot{p1, p2, p3, p4}
}

type statsPlot struct {
	*qml.Plot
	name   []string
	keys   []string
	x, y   []*vec.Vector
	config []network.Config
}

// clear data
func (p *statsPlot) clear() {
	p.name = []string{}
	p.keys = []string{}
	p.x = []*vec.Vector{}
	p.y = []*vec.Vector{}
	p.config = []network.Config{}
	p.Clear()
}

// add distinct key
func (p *statsPlot) addKey(key string) {
	for _, k := range p.keys {
		if k == key {
			return
		}
	}
	p.keys = append(p.keys, key)
	return
}

// lookup title: if not found then add it
func (p *statsPlot) find(cfg *network.Config) (ix int, ok bool) {
	// already have a plot with these settings?
	for i, c := range p.config {
		if *cfg == c {
			return i, true
		}
	}
	// add new plot
	ix = len(p.name)
	p.config = append(p.config, *cfg)
	p.x = append(p.x, vec.New(0))
	p.y = append(p.y, vec.New(0))
	// get legend text
	p.name = append(p.name, "")
	if ix > 0 {
		def := &p.config[0]
		for _, key := range config.Keys(cfg) {
			if config.Get(def, key) != config.Get(cfg, key) {
				p.addKey(key)
			}
		}
		for i, conf := range p.config {
			p.name[i] = ""
			for _, key := range p.keys {
				p.name[i] += fmt.Sprintf("%s=%v ", key, config.Get(&conf, key))
			}
		}
	}
	return
}

// add point to the stats plot
func (p *statsPlot) addPoint(s *network.Stats, cfg *network.Config) {
	i, found := p.find(cfg)
	p.x[i].Push(s.RunTime.Mean, s.RunTime.StdDev)
	p.y[i].Push(s.ClsError.Mean, s.ClsError.StdDev)
	if !found {
		for i, title := range p.name[:len(p.name)-1] {
			p.Plotters[i].SetName(title)
		}
		pts := qml.NewPoints(p.x[i], p.y[i], p.name[i])
		pts.Color = qml.Color(i)
		pts.DrawErrors = true
		p.Add(pts)
	}
}

// train the network
func train(cfg *network.Config, net *network.Network, data *data.Dataset, s *network.Stats, ctrl *qml.Ctrl, p *statsPlot) {
	var running, started bool
	var stopCond func(*network.Stats) (bool, bool)
	run := 0
	p.clear()

	startRun := func() {
		// start new training run
		ctrl.SetRun(run + 1)
		net.SetRandomWeights()
		stopCond = network.StopCriteria(cfg)
		if run == 0 {
			s.Reset()
		}
		s.StartRun()
		started = true
	}

	endRun := func(failed bool) {
		// print stats at end of run
		run++
		fmt.Println(s.EndRun(failed))
		started = false
		if run >= cfg.MaxRuns {
			running = false
			ctrl.Done()
			fmt.Printf("%s\n\n", s.History())
			p.addPoint(s, cfg)
			run = 0
		}
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
			if done, failed := stopCond(s); done {
				endRun(failed)
			}
		case "run": // toggle running mode
			running = (ev.Arg == "start")
		case "stats": // print out the stats over runs
			if ev.Arg == "print" {
				fmt.Printf("%s\n\n", s.History())
			} else if ev.Arg == "clear" {
				p.clear()
				s.Reset()
			}
		case "select": // choose a new data set
			p.clear()
			s.Reset()
			cfg, net, data, _ = network.Load(ev.Arg)
			cfg.Print()
			ctrl.Refresh(cfg)
			running = false
			started = false
			run = 0
		case "quit": // exit the program
			net.Release()
			blas.Release()
			ctrl.WG.Done()
			return
		}
	}
}
