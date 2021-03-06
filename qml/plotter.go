package qml

import (
	"gopkg.in/qml.v1/gl/2.1"
	"image/color"
)

const (
	legendWidth      = 0.03
	legendPadding    = 0.005
	defaultLineWidth = 2
	defaultPointSize = 5
)

// Plotter interface type is a plottable data set.
type Plotter interface {
	Refresh()
	Plot(gl *GL.GL, p *Plot)
	LegendSize(ps *Plots) (x, y float32)
	DrawLegend(gl *GL.GL, ps *Plots)
	DataRange() (min, max Point)
	SetColor(c color.RGBA)
	SetName(n string)
	Scaled() bool
}

type XYer interface {
	Len() int
	BinWidth() float32
	XY(i int) (x, y float32)
	XYErr(i int) (x, y, yerr float32)
	DataRange() (xmin, ymin, xmax, ymax float32)
	Lock()
	Unlock()
}

type Point struct {
	X, Y float32
}

func NewPoint(x, y float32) Point {
	return Point{x, y}
}

// base plotter type
type plotter struct {
	Name  string
	Color color.RGBA
	pts   []Point
	min   Point
	max   Point
}

func newPlotter(name string) *plotter {
	return &plotter{
		Name:  name,
		Color: DefaultColors[0],
	}
}

func (pt *plotter) Scaled() bool { return false }

func (pt *plotter) SetColor(c color.RGBA) { pt.Color = c }

func (pt *plotter) SetName(n string) { pt.Name = n }

// DataRange method gives the x and y range
func (pt *plotter) DataRange() (min, max Point) {
	return pt.min, pt.max
}

// LegendSize method returns the size of the legend on screen.
func (pt *plotter) LegendSize(ps *Plots) (w, h float32) {
	if pt.Name != "" {
		w = 3*legendPadding + legendWidth + ps.TextWidth(pt.Name)
		h = legendPadding + ps.TextHeight()
	}
	return
}

// DrawLegend method draws the legend at the current screen position.
func (pt *plotter) DrawLegend(gl *GL.GL, ps *Plots) {
	if pt.Name != "" {
		// draw a line
		gl.PushAttrib(GL.CURRENT_BIT)
		setColor(gl, pt.Color)
		ypos := legendPadding + ps.TextHeight()/2
		drawLine(gl, legendPadding, -ypos, legendPadding+legendWidth, -ypos)
		gl.PopAttrib()
		// draw a label
		ps.font.DrawText(gl, 2*legendPadding+legendWidth, -legendPadding, 0, pt.Name)
	}
}

// Line plotter represents a line plot.
type Line struct {
	*plotter
	data  XYer
	Width float32
}

// NewLine function creates a new line with the given data
func NewLine(pts XYer, name string) *Line {
	return &Line{
		plotter: newPlotter(name),
		data:    pts,
		Width:   defaultLineWidth,
	}
}

// Plot method implements the plotter interface
func (l *Line) Plot(gl *GL.GL, p *Plot) {
	if len(l.pts) < 2 {
		return
	}
	gl.PushAttrib(GL.CURRENT_BIT | GL.LINE_BIT)
	gl.LineWidth(l.Width)
	setColor(gl, l.Color)
	gl.PushMatrix()
	p.rescale(gl)
	gl.Begin(GL.LINE_STRIP)
	for _, pt := range l.pts {
		gl.Vertex2f(pt.X, pt.Y)
	}
	gl.End()
	gl.PopMatrix()
	gl.PopAttrib()
}

// Refresh method is called to update the data
func (l *Line) Refresh() {
	l.data.Lock()
	defer l.data.Unlock()
	l.pts = make([]Point, l.data.Len())
	for i := range l.pts {
		x, y := l.data.XY(i)
		l.pts[i] = NewPoint(x, y)
	}
	x0, y0, x1, y1 := l.data.DataRange()
	l.min = NewPoint(x0, y0)
	l.max = NewPoint(x1, y1)
}

// Points plotter represents an xy scatter plot
type Points struct {
	*plotter
	errors     []Point
	xdata      XYer
	ydata      XYer
	PointSize  float32
	DrawErrors bool
}

// NewPoints function creates a new scatter plot
func NewPoints(xpts, ypts XYer, name string) *Points {
	return &Points{
		plotter:   newPlotter(name),
		xdata:     xpts,
		ydata:     ypts,
		PointSize: defaultPointSize,
	}
}

// Plot method implements the plotter interface
func (d *Points) Plot(gl *GL.GL, p *Plot) {
	if len(d.pts) < 1 {
		return
	}
	gl.PushAttrib(GL.CURRENT_BIT)
	setColor(gl, d.Color)
	gl.PushMatrix()
	p.rescale(gl)
	if d.DrawErrors {
		// draw with x and y error bars
		for i, pt := range d.pts {
			err := d.errors[i]
			drawLine(gl, pt.X-err.X, pt.Y, pt.X+err.X, pt.Y)
			drawLine(gl, pt.X, pt.Y-err.Y, pt.X, pt.Y+err.Y)
		}
	} else {
		// draw as dots
		gl.PointSize(d.PointSize)
		gl.Enable(GL.BLEND)
		gl.BlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA)
		gl.Enable(GL.POINT_SMOOTH)
		gl.Begin(GL.POINTS)
		for _, pt := range d.pts {
			gl.Vertex2f(pt.X, pt.Y)
		}
		gl.End()
	}
	gl.PopMatrix()
	gl.PopAttrib()
}

// Refresh method is called to update the data
func (d *Points) Refresh() {
	d.xdata.Lock()
	d.ydata.Lock()
	defer d.xdata.Unlock()
	defer d.ydata.Unlock()
	d.pts = make([]Point, d.xdata.Len())
	d.errors = make([]Point, d.xdata.Len())
	for i := range d.pts {
		_, x, xerr := d.xdata.XYErr(i)
		_, y, yerr := d.ydata.XYErr(i)
		d.pts[i] = NewPoint(x, y)
		d.errors[i] = NewPoint(xerr, yerr)
	}
	_, x0, _, x1 := d.xdata.DataRange()
	_, y0, _, y1 := d.ydata.DataRange()
	d.min = NewPoint(x0, y0)
	d.max = NewPoint(x1, y1)
}

// Histogram plotter represents a histogram plot.
type Histogram struct {
	*plotter
	data XYer
	bins []bin
	Fill bool
}

// NewHistogram function creates a histogram plot with the given data
func NewHistogram(pts XYer, name string) *Histogram {
	return &Histogram{
		plotter: newPlotter(name),
		data:    pts,
		bins:    []bin{},
		Fill:    true,
	}
}

func (h *Histogram) Scaled() bool { return true }

// Plot method implements the plotter interface
func (h *Histogram) Plot(gl *GL.GL, p *Plot) {
	if len(h.bins) < 1 {
		return
	}
	gl.PushAttrib(GL.CURRENT_BIT)
	gl.PushMatrix()
	p.rescale(gl)
	// fill with transparency
	if h.Fill {
		fill := h.Color
		fill.A = 16
		setColor(gl, fill)
		gl.PushAttrib(GL.ENABLE_BIT)
		gl.Enable(GL.BLEND)
		gl.BlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA)
		for _, b := range h.bins {
			if b.val > 0 {
				drawBox(gl, b.min, 0, b.max, b.val, true)
			}
		}
		gl.PopAttrib()
	}
	// draw outline
	setColor(gl, h.Color)
	for _, b := range h.bins {
		if b.val > 0 {
			drawBox(gl, b.min, 0, b.max, b.val, false)
		}
	}
	gl.PopMatrix()
	gl.PopAttrib()
}

// Refresh method is called to update the data
func (h *Histogram) Refresh() {
	h.data.Lock()
	defer h.data.Unlock()
	h.bins = make([]bin, h.data.Len())
	w := h.data.BinWidth()
	for i := range h.bins {
		x, y := h.data.XY(i)
		h.bins[i] = bin{min: float32(x), max: float32(x + w), val: float32(y)}
	}
	if h.data.Len() > 0 {
		x0, y0, x1, y1 := h.data.DataRange()
		h.min = NewPoint(x0, y0)
		h.max = NewPoint(x1, y1)
	}
}

type bin struct {
	min, max, val float32
}
