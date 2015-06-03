// Package qml implements a GUI interface and plotting functions using the go QML bindings.
package qml

import (
	"bytes"
	"github.com/jnb666/deepthought/config"
	"github.com/jnb666/deepthought/network"
	"go/build"
	"gopkg.in/qml.v1"
	"reflect"
	"strings"
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
	conf     *Config
	WG       sync.WaitGroup
	ev       chan Event
	run      qml.Object
	plot     qml.Object
	setNames []string
	current  int
}

func NewCtrl(cfg *network.Config, dataSets []string, selected string) *Ctrl {
	c := new(Ctrl)
	c.conf = &Config{cfg: cfg}
	c.setNames = dataSets
	for ix, set := range dataSets {
		if strings.ToLower(set) == strings.ToLower(selected) {
			c.current = ix
		}
	}
	c.ev = make(chan Event, 10)
	c.WG.Add(1)
	return c
}

func (c *Ctrl) init(root qml.Object) {
	c.run = root.ObjectByName("runButton")
	c.plot = root.ObjectByName("plotControl")
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
func (c *Ctrl) Select(index int) {
	if index == c.current {
		return
	}
	c.current = index
	c.ev <- Event{"select", c.setNames[index]}
}

// Callback when run is completed
func (c *Ctrl) Done() {
	qml.RunMain(func() { c.run.Set("checked", false) })
}

// Callback to refresh the plot
func (c *Ctrl) Refresh(cfg *network.Config) {
	c.conf.cfg = cfg
	qml.RunMain(func() {
		c.plot.Call("update")
		c.conf.Update()
	})
}

// Config type manages updating the config settings
type Config struct {
	cfg   *network.Config
	opts  []qml.Object
	name  []string
	ready bool
}

func (c *Config) init(root qml.Object) {
	s := reflect.ValueOf(c.cfg).Elem()
	c.opts = make([]qml.Object, s.NumField())
	c.name = make([]string, s.NumField())
	for i := range c.opts {
		name := s.Type().Field(i).Name
		opt := root.ObjectByName(name)
		if name == "Sampler" {
			opt.On("activated", func() {
				config.Set(c.cfg, name, network.SamplerNames[opt.Int("currentIndex")])
			})

		} else {
			opt.On("editingFinished", func() {
				config.Set(c.cfg, name, opt.String("text"))
			})
		}
		c.opts[i] = opt
		c.name[i] = name
	}
	c.ready = true
}

// Update display of config settings
func (c *Config) Update() {
	if !c.ready {
		return
	}
	for i, opt := range c.opts {
		value := config.Get(c.cfg, c.name[i])
		if c.name[i] == "Sampler" {
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

// Print current config settings to stdout
func (c *Config) Print() {
	config.Print(c.cfg)
}

// MainLoop function is called to draw the scene, it does not return.
// ctrl is the control structure used for sending events and plts is a list of plots to display.
func MainLoop(ctrl *Ctrl, plts ...*Plot) {
	if len(plts) < 1 {
		panic("must provide at least one plot to display")
	}
	err := qml.Run(func() error {
		qml.RegisterTypes("GoExtensions", 1, 0, []qml.TypeSpec{{
			Init: func(p *Plots, obj qml.Object) {
				p.Object = obj
				p.plt = plts
			},
		}})
		engine := qml.NewEngine()
		scene := getScene(plts, ctrl.setNames, ctrl.current)
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
		tabs.Set("currentIndex", 1)
		ctrl.conf.init(tabs.Call("getTab", 1).(qml.Object))
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
func getScene(plts []*Plot, names []string, ix int) string {
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
	var buf bytes.Buffer
	err = tmpl.Execute(&buf, params{Plots: ps, Datasets: names, Index: ix})
	if err != nil {
		panic(err)
	}
	return buf.String()
}
