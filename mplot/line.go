package mplot

import (
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg/draw"
	"image/color"
	"math"
)

var DefaultColors = []color.Color{
	rgb(255, 255, 0),
	rgb(0, 255, 255),
	rgb(255, 105, 180),
	rgb(0, 255, 0),
	rgb(255, 0, 255),
	rgb(0, 0, 255),
	rgb(255, 0, 0),
}

func rgb(r, g, b uint8) color.RGBA {
	return color.RGBA{r, g, b, 255}
}

// Color returns the ith default color, wrapping
// if i is less than zero or greater than the max
// number of colors in the DefaultColors slice.
func Color(i int) color.Color {
	n := len(DefaultColors)
	if i < 0 {
		return DefaultColors[i%n+n]
	}
	return DefaultColors[i%n]
}

// Line implements the Plotter interface, drawing a line.
type Line struct {
	*Vector
	draw.LineStyle
	Name string
}

// NewLine function creates a new line with default line style. Input is a column vector and range of data.
func NewLine(vec *Vector, name string) *Line {
	return &Line{
		Vector:    vec,
		LineStyle: plotter.DefaultLineStyle,
		Name:      name,
	}
}

// Plot draws the Line, implementing the plot.Plotter interface.
func (l *Line) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, trY := plt.Transforms(&c)
	ps := make([]draw.Point, l.Len())
	for i := range ps {
		x, y := l.XY(i)
		ps[i].X = trX(x)
		ps[i].Y = trY(y)
	}
	c.StrokeLines(l.LineStyle, c.ClipLinesXY(ps)...)
}

// DataRange implements the plot.DataRanger interface.
func (l *Line) DataRange() (xmin, xmax, ymin, ymax float64) {
	xmin, xmax, ymin = 0, float64(l.Cap()-1), 0
	yval := l.max
	if l.Len() > 1 && 2*l.Last() < yval {
		yval = 2 * l.Last()
	}
	ymax = math.Pow(2, math.Ceil(math.Log2(yval)))
	return
}

// Thumbnail implements the plot.Thumbnailer interface.
func (l *Line) Thumbnail(c *draw.Canvas) {
	y := c.Center().Y
	c.StrokeLine2(l.LineStyle, c.Min.X, y, c.Max.X, y)
}

// Add function adds new lines to the plot picking a colour from the palette.
func AddLines(plt *Plot, lines ...*Line) {
	ps := make([]plot.Plotter, len(lines))
	for i, line := range lines {
		line.Color = Color(i)
		ps[i] = line
		if line.Name != "" {
			plt.Legend.Add(line.Name, line)
		}
		plt.ranges = append(plt.ranges, line)
	}
	plt.Add(ps...)
}
