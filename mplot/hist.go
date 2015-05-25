package mplot

import (
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg/draw"
	"github.com/jnb666/deepthought/blas"
	"image/color"
	"math"
)

// Histogram implements the Plotter interface, drawing a histogram.
type Histogram struct {
	blas.Matrix
	FillColor  color.Color
	Buffer     int
	bins       []bin
	inSync     bool
	xmin, xmax float64
	ymax       float64
}

type bin struct {
	min, max, val float64
}

// NewHistogram function creates a new histogram using the buf column vector.
func NewHistogram(nbins int, min, max float64) *Histogram {
	return &Histogram{
		FillColor: color.Gray{128},
		Matrix:    blas.New(nbins, 2),
		bins:      makeBins(nbins, min, max),
		xmin:      min,
		xmax:      max,
		ymax:      10,
	}
}

// Add method adds samples from a column vector
func (h *Histogram) Update(vec blas.Matrix, buf int) {
	h.Col(buf, buf+1).Histogram(vec, len(h.bins), h.xmin, h.xmax)
	h.inSync = false
}

// Plot implements the Plotter interface, drawing a box for each bin.
func (h *Histogram) Plot(c draw.Canvas, p *plot.Plot) {
	trX, trY := p.Transforms(&c)
	if !h.inSync {
		h.refresh()
	}
	for _, bin := range h.bins {
		pts := []draw.Point{
			{trX(bin.min), trY(0)},
			{trX(bin.max), trY(0)},
			{trX(bin.max), trY(bin.val)},
			{trX(bin.min), trY(bin.val)},
		}
		if h.FillColor != nil {
			c.FillPolygon(h.FillColor, c.ClipPolygonXY(pts))
		}
		pts = append(pts, draw.Point{trX(bin.min), trY(0)})
		c.StrokeLines(plotter.DefaultLineStyle, c.ClipLinesXY(pts)...)
	}
}

// DataRange returns the minimum and maximum X and Y values
func (h *Histogram) DataRange() (xmin, xmax, ymin, ymax float64) {
	if !h.inSync {
		h.refresh()
	}
	xmin, xmax, ymin = h.xmin, h.xmax, 0
	ymax = math.Pow(2, math.Ceil(math.Log2(h.ymax)))
	return
}

// divide range into bins
func makeBins(nbins int, xmin, xmax float64) []bin {
	b := make([]bin, nbins)
	width := (xmax - xmin) / float64(nbins)
	x := xmin
	for i := range b {
		b[i] = bin{min: x, max: x + width}
		x += width
	}
	return b
}

// refresh the data
func (h *Histogram) refresh() {
	h.ymax = 0
	data := h.Col(h.Buffer, h.Buffer+1).Data(blas.ColMajor)
	for i := range h.bins {
		h.bins[i].val = data[i]
		if data[i] > h.ymax {
			h.ymax = data[i]
		}
	}
	h.inSync = true
}

// Add function adds new histograms to the plot picking a colour from the palette.
func AddHist(plt *Plot, hists ...*Histogram) {
	ps := make([]plot.Plotter, len(hists))
	for i, hist := range hists {
		hist.FillColor = Color(i)
		ps[i] = hist
		plt.ranges = append(plt.ranges, hist)
	}
	plt.Add(ps...)
}
