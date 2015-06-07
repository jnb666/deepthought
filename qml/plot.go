package qml

import (
	"gopkg.in/qml.v1"
	"gopkg.in/qml.v1/gl/2.1"
	"image/color"
)

type Position int

const (
	TopRight Position = iota
	TopLeft
	BottomRight
	BottomLeft
)

const (
	xmin, ymin = -0.85, 0.85
	xmax, ymax = 0.95, -0.92
	big        = float32(1e30)
	epsilon    = 1e-10
)

var DefaultColors = []color.RGBA{
	rgb(255, 255, 0),
	rgb(0, 255, 255),
	rgb(255, 105, 180),
	rgb(0, 255, 0),
	rgb(0, 0, 255),
	rgb(255, 0, 0),
	rgb(255, 0, 255),
}

func rgb(r, g, b uint8) color.RGBA {
	return color.RGBA{r, g, b, 255}
}

// Color returns the ith default color
func Color(i int) color.RGBA {
	n := len(DefaultColors)
	if i < 0 {
		return DefaultColors[i%n+n]
	}
	return DefaultColors[i%n]
}

// Plots type is a QML plotting component.
type Plots struct {
	qml.Object
	Grid       bool
	Background color.RGBA
	Color      color.RGBA
	GridColor  color.RGBA
	plt        []*Plot
	current    int
	font       *Font
	width      int
	height     int
}

// Plot type represents one of the plots
type Plot struct {
	Name     string
	Title    string
	Xaxis    *Axis
	Yaxis    *Axis
	Legend   Position
	Plotters []Plotter
}

// NewPlot function creates a new plot with default settings
func NewPlot(name, title string, plt ...Plotter) *Plot {
	p := &Plot{Name: name, Title: title}
	p.Xaxis = newAxis(true)
	p.Yaxis = newAxis(false)
	p.Plotters = make([]Plotter, len(plt))
	for i, element := range plt {
		p.Plotters[i] = element
		p.Plotters[i].SetColor(Color(i))
	}
	return p
}

// Add method adds elements to the plot
func (p *Plot) Add(plt ...Plotter) {
	p.Plotters = append(p.Plotters, plt...)
}

// Clear method removes all the ploters from the plot
func (p *Plot) Clear() {
	p.Plotters = []Plotter{}
}

// Switch the plot to display
func (ps *Plots) Select(i int) {
	ps.current = i
	ps.Call("update")
}

var prevxmax = float32(0)

// Paint method draws the plot.
func (ps *Plots) Paint(paint *qml.Painter) {
	var err error
	gl := GL.API(paint)
	ps.width, ps.height = ps.Int("width"), ps.Int("height")
	if ps.font == nil {
		ps.font, err = ps.loadFont(gl, DefaultFontName, DefaultFontScale)
		if err != nil {
			panic(err)
		}
	}
	// set screen transform
	gl.Viewport(0, 0, ps.width, ps.height)
	gl.MatrixMode(GL.PROJECTION)
	gl.LoadIdentity()
	gl.Ortho(-1, 1, -1, 1, -1, 1)
	gl.MatrixMode(GL.MODELVIEW)
	gl.LoadIdentity()
	gl.Translatef(xmin, ymin, 0)
	gl.Scalef(xmax-xmin, ymax-ymin, 1)
	// set background colour
	r, g, b, a := col(ps.Background)
	gl.ClearColor(r, g, b, a)
	gl.Clear(GL.COLOR_BUFFER_BIT)
	// refresh the plot data
	p := ps.plt[ps.current]
	for _, plt := range p.Plotters {
		plt.Refresh()
	}
	// draw the axes
	ps.drawAxes(gl, p)
	// draw the plotters
	for _, plt := range p.Plotters {
		plt.Plot(gl, p)
	}
	// draw the legend
	ps.drawLegend(gl, p)
	// draw the plot title
	ps.font.DrawText(gl, 0.5-ps.TextWidth(p.Title)/2, 1+ps.TextHeight(), 0, p.Title)
}

func (ps *Plots) drawLegend(gl *GL.GL, p *Plot) {
	var width, height float32
	for _, plt := range p.Plotters {
		w, h := plt.LegendSize(ps)
		width = fmax(width, w)
		height = fmax(height, h)
	}
	ypos := float32(1)
	if p.Legend == BottomLeft || p.Legend == BottomRight {
		ypos = height * float32(len(p.Plotters))
	}
	for _, plt := range p.Plotters {
		gl.PushMatrix()
		if p.Legend == TopLeft || p.Legend == BottomLeft {
			gl.Translatef(0, ypos, 0)
		} else {
			gl.Translatef(1-width, ypos, 0)
		}
		plt.DrawLegend(gl, ps)
		gl.PopMatrix()
		ypos -= height
	}
}

// Trans method converts from plot to screen coordinates
func (p *Plot) Trans(x0, y0 float32) (x1, y1 float32) {
	x1 = (x0 - p.Xaxis.min) / (p.Xaxis.max - p.Xaxis.min)
	y1 = (y0 - p.Yaxis.min) / (p.Yaxis.max - p.Yaxis.min)
	return
}

// rescale method adjusts the transform from screen to plot coordinates
func (p *Plot) rescale(gl *GL.GL) {
	gl.Scalef(1/(p.Xaxis.max-p.Xaxis.min), 1/(p.Yaxis.max-p.Yaxis.min), 1)
	gl.Translatef(-p.Xaxis.min, -p.Yaxis.min, 0)
}

// utilities
func col(c color.RGBA) (r, g, b, a float32) {
	return float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255, float32(c.A) / 255
}

func setColor(gl *GL.GL, c color.RGBA) {
	r, g, b, a := col(c)
	gl.Color4f(r, g, b, a)
}

func drawLine(gl *GL.GL, x0, y0, x1, y1 float32) {
	gl.Begin(GL.LINES)
	gl.Vertex2f(x0, y0)
	gl.Vertex2f(x1, y1)
	gl.End()
}

func fmin(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func fmax(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
