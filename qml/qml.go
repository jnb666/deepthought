// Package qml implements a GUI interface and plotting functions using the go QML bindings.
package qml

import (
	"bytes"
	"fmt"
	"github.com/jnb666/deepthought/config"
	"github.com/jnb666/deepthought/network"
	"go/build"
	"gopkg.in/qml.v1"
	"sync"
	"text/template"
)

const importString = "github.com/jnb666/deepthought/qml"

type Event struct {
	Typ string
	Arg string
}

// Ctrl type is used for communication with the gui
type Ctrl struct {
	conf      *Config
	network   *network.Network
	testData  *network.Data
	WG        sync.WaitGroup
	ev        chan Event
	net       qml.Object
	run       qml.Object
	plot      qml.Object
	runLabel  qml.Object
	testLabel qml.Object
	filter    qml.Object
	distort   qml.Object
	setNames  []string
	plots     []*Plot
}

func NewCtrl(cfg *network.Config, net *network.Network, testData *network.Data, dataSets []string, selected string, plots []*Plot) *Ctrl {
	c := new(Ctrl)
	c.conf = &Config{cfg: cfg, Model: selected}
	c.network = net
	c.testData = testData
	c.setNames = dataSets
	c.ev = make(chan Event, 10)
	c.plots = plots
	c.WG.Add(1)
	return c
}

func (c *Ctrl) init(root qml.Object) {
	c.run = root.ObjectByName("runButton")
	c.plot = root.ObjectByName("plotControl")
	c.runLabel = root.ObjectByName("runLabel")
	c.runLabel.Set("text", fmt.Sprintf("run: 1/%d", c.conf.cfg.MaxRuns))
}

func (c *Ctrl) initNet(root qml.Object) {
	c.net = root.ObjectByName("netControl")
	c.testLabel = root.ObjectByName("testLabel")
	c.filter = root.ObjectByName("filterList")
	c.distort = root.ObjectByName("distortList")
	c.setComboLists()
	c.net.Call("run", 1)
}

// set filter and distort combo box lists
func (c *Ctrl) setComboLists() {
	c.filter.Call("reset")
	c.filter.Call("addItem", "any")
	nout := c.testData.Output.Cols()
	if nout == 1 {
		nout = 2
	}
	for i := 0; i < nout; i++ {
		c.filter.Call("addItem", fmt.Sprint(i))
	}
	c.distort.Call("reset")
	c.distort.Call("addItem", "all")
	for _, t := range getLoader(c.conf.Model).DistortTypes() {
		c.distort.Call("addItem", t.Name)
	}
}

// Get next event from channel, if running is set then auto step
func (c *Ctrl) NextEvent(running bool) Event {
	if running {
		select {
		case e := <-c.ev:
			return e
		default:
			return Event{Typ: "step"}
		}
	}
	return <-c.ev
}

// Send user input invent
func (c *Ctrl) Send(typ, arg string) {
	c.ev <- Event{typ, arg}
}

// Choose new data set
func (c *Ctrl) Select(model string) {
	if model != c.conf.Model {
		c.conf.Model = model
		c.ev <- Event{"select", model}
	}
}

// Callback when run is completed
func (c *Ctrl) Done() {
	qml.RunMain(func() { c.run.Set("checked", false) })
}

// Callback to refresh the plot when selecting a new dataset
func (c *Ctrl) Refresh(cfg *network.Config, net *network.Network, testData *network.Data) {
	qml.RunMain(func() {
		c.conf.cfg = cfg
		c.conf.Update()
		c.plot.Call("update")
		c.network = net
		c.testData = testData
		c.setComboLists()
		c.net.Call("first")
	})
}

// Callback to set run number
func (c *Ctrl) SetRun(run int) {
	qml.RunMain(func() {
		label := fmt.Sprintf("run: %d/%d", run, c.conf.cfg.MaxRuns)
		c.runLabel.Set("text", label)
		c.net.Call("first")
	})
}

// Config type manages updating the config settings
type Config struct {
	Model  string
	cfg    *network.Config
	opts   []qml.Object
	keys   []string
	loader network.Loader
}

// initialise options struct
func (c *Config) init(root qml.Object) {
	c.keys = config.Keys(c.cfg)
	c.opts = make([]qml.Object, len(c.keys))
	for i, key := range c.keys {
		c.opts[i] = root.ObjectByName(key)
	}
	c.Update()
}

// Set config value from GUI
func (c *Config) Set(key, value string) {
	config.Set(c.cfg, key, value)
}

// Update display of config settings
func (c *Config) Update() {
	for i, opt := range c.opts {
		value := config.Get(c.cfg, c.keys[i])
		if c.keys[i] == "Sampler" {
			for i, name := range network.SamplerNames {
				if name == value {
					opt.Set("currentIndex", i)
				}
			}
		} else {
			opt.Set("text", value)
		}
	}
}

// Set default config settings
func (c *Config) Default(model string) {
	config.Update(c.cfg, getLoader(model).Config())
	c.Update()
}

func (c *Config) Print() {
	config.Print(c.cfg)
}

// Save current config settings to disk
func (c *Config) Save(model string) {
	if err := config.Save(c.cfg, model); err != nil {
		fmt.Println("error saving config:", err)
	}
}

// Load config settings from disk
func (c *Config) Load(model string) {
	if err := config.Load(c.cfg, model); err != nil {
		fmt.Println("error loading config")
	}
	c.Update()
}

// MainLoop function is called to draw the scene, it does not return.
// ctrl is the control structure used for sending events and plts is a list of plots to display.
func MainLoop(ctrl *Ctrl) {
	err := qml.Run(func() error {
		qml.RegisterTypes("GoExtensions", 1, 0, []qml.TypeSpec{
			{
				Init: func(p *Plots, obj qml.Object) {
					p.Object = obj
					p.plt = ctrl.plots
				},
			},
			{
				Init: func(n *Network, obj qml.Object) {
					n.Object = obj
					n.ctrl = ctrl
					n.filter = -1
				},
			},
		})
		engine := qml.NewEngine()
		scene := getScene(ctrl.plots, ctrl.setNames, ctrl.conf.Model)
		component, err := engine.LoadString("plot", scene)
		if err != nil {
			return err
		}
		context := engine.Context()
		context.SetVar("ctrl", ctrl)
		context.SetVar("cfg", ctrl.conf)
		win := component.CreateWindow(nil)
		root := win.Root()
		tabs := root.ObjectByName("tabs")
		tabs.Set("currentIndex", 2)
		ctrl.conf.init(tabs.Call("getTab", 2).(qml.Object))
		tabs.Set("currentIndex", 1)
		ctrl.initNet(tabs.Call("getTab", 1).(qml.Object))
		tabs.Set("currentIndex", 0)
		ctrl.init(tabs.Call("getTab", 0).(qml.Object))
		win.Show()
		win.Wait()
		return nil
	})
	if err != nil {
		panic(err)
	}
}

type params struct {
	Plots    []plotSel
	Datasets []string
	Index    int
}

type plotSel struct {
	Name     string
	Index    int
	Selected bool
}

// construct the scene description
func getScene(plts []*Plot, names []string, model string) string {
	pkg, err := build.Import(importString, "", build.FindOnly)
	if err != nil {
		panic("cannot find package " + importString)
	}
	tmpl, err := template.ParseFiles(pkg.Dir + "/main.qml")
	if err != nil {
		panic(err)
	}
	ps := make([]plotSel, len(plts))
	for i, p := range plts {
		ps[i].Name = p.Name
		ps[i].Index = i
		ps[i].Selected = (i == 0)
	}
	ix := 0
	for i := range names {
		if names[i] == model {
			ix = i
		}
	}
	var buf bytes.Buffer
	err = tmpl.Execute(&buf, params{Plots: ps, Datasets: names, Index: ix})
	if err != nil {
		panic(err)
	}
	return buf.String()
}
