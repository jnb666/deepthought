// Package qml implements a GUI interface and plotting functions using the go QML bindings.
package qml

import (
	"bytes"
	"gopkg.in/qml.v1"
	"sync"
	"text/template"
)

type Event struct {
	Typ string
	Arg string
}

// Ctrl type is used for communication with the gui
type Ctrl struct {
	WG       sync.WaitGroup
	ev       chan Event
	run      qml.Object
	plot     qml.Object
	setNames []string
	current  int
}

func NewCtrl(dataSets []string, selected int) *Ctrl {
	c := new(Ctrl)
	c.setNames = dataSets
	c.current = selected
	c.ev = make(chan Event, 10)
	c.WG.Add(1)
	return c
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
	if c.run != nil {
		qml.RunMain(func() { c.run.Set("checked", false) })
	}
}

// Callback to refresh the plot
func (c *Ctrl) Refresh() {
	if c.plot != nil {
		qml.RunMain(func() { c.plot.Call("update") })
	}
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
		win := component.CreateWindow(nil)
		ctrl.run = win.Root().ObjectByName("runButton")
		ctrl.plot = win.Root().ObjectByName("plotControl")
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
	tmpl, err := template.New("scene").Parse(sceneTmpl)
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

var sceneTmpl = `
import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.1
import GoExtensions 1.0

ApplicationWindow {
	id: root; title: "deepthought"; color: "lightgray"; x: 50; y: 50
	onClosing: ctrl.send("quit", "")

	ColumnLayout {	
		RowLayout {
			spacing: 20
			Button { text: "restart"; onClicked: ctrl.send("start", "") }
			Button { text: "step"; onClicked: ctrl.send("step", "") }
			Button {
				objectName: "runButton"; text: "run"; checkable: true
				onClicked: ctrl.send("run", checked ? "start" : "stop");
			}
			Button { text: "stats"; onClicked: ctrl.send("stats", "") }
			Label { text: "data set:" }
			ComboBox {
				objectName: "modelList"
				width: 120
				model: ListModel {
					{{ range .Datasets }}
					ListElement { text: "{{ . }}" }
					{{ end }}
				}
				currentIndex: {{ .Index }}
				onActivated: ctrl.select(index)
			}
		}
		Plots{
			id: plot; objectName: "plotControl"
			width: 800; height: 800
			background: "black"; color: "white"
			grid: true; gridColor: "#404040"
			Timer {
				interval: 100; running: true; repeat: true
				onTriggered: plot.update()
			}
		}
		RowLayout {
			ExclusiveGroup { id: plotGroup }
			{{ range .Plots }}
			RadioButton { 
				text: "{{ .Name }}"; exclusiveGroup: plotGroup; checked: {{ .Selected }}
				onClicked: plot.select({{ .Index }})
			}
			{{ end }}
		}
	}
}
`
