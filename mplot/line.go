package mplot

import (
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg/draw"
	"github.com/jnb666/deepthought/m32"
)

// Line implements the Plotter interface, drawing a line.
type Line struct {
	draw.LineStyle
	Name     string
	data     *m32.Matrix
	min, max float64
}

// NewLine function creates a new line with default line style. Input is a column vector and range of data.
func NewLine(vec *m32.Matrix, name string, ymin, ymax float64) *Line {
	return &Line{
		LineStyle: plotter.DefaultLineStyle,
		Name:      name,
		data:      vec,
		min:       ymin,
		max:       ymax,
	}
}

// Plot draws the Line, implementing the plot.Plotter interface.
func (l *Line) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	ps := make([]draw.Point, l.data.Rows)
	for i := range ps {
		ps[i].X = trX(float64(i))
		ps[i].Y = trY(float64(l.data.At(i, 0)))
	}
	c.StrokeLines(l.LineStyle, c.ClipLinesXY(ps)...)
}

// DataRange implements the plot.DataRanger interface.
func (l *Line) DataRange() (xmin, xmax, ymin, ymax float64) {
	return 0, float64(l.data.Size() - 1), l.min, l.max
}

// Thumbnail implements the plot.Thumbnailer interface.
func (l *Line) Thumbnail(c *draw.Canvas) {
	y := c.Center().Y
	c.StrokeLine2(l.LineStyle, c.Min.X, y, c.Max.X, y)
}

// Add function adds new lines to the plot.
func AddLines(plt *plot.Plot, lines ...*Line) {
	ps := make([]plot.Plotter, len(lines))
	for i, line := range lines {
		line.Color = plotutil.Color(i)
		ps[i] = line
		plt.Legend.Add(line.Name, line)
	}
	plt.Add(ps...)
}
