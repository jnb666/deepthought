package mplot

import (
	//"fmt"
	"github.com/gonum/plot"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/jnb666/deepthought/blas"
	"image/color"
)

// Image implements the Plotter interface, drawing matrix as a grayscale image
type Image struct {
	blas.Matrix
}

// NewImage function creates a new image from a matrix.
func NewImage(m blas.Matrix) *Image {
	return &Image{Matrix: m}
}

// Plot draws the Image, implementing the plot.Plotter interface.
func (img *Image) Plot(c draw.Canvas, plt *plot.Plot) {
	data := img.Data(blas.RowMajor)
	trX, trY := plt.Transforms(&c)
	j := 0
	pts := make([]draw.Point, 4)
	rows, cols := img.Rows(), img.Cols()
	var x0, y0, x1, y1 vg.Length
	for iy := 0; iy < rows; iy++ {
		y0, y1 = trY(float64(rows-iy-1)), trY(float64(rows-iy))+1
		for ix := 0; ix < cols; ix++ {
			x0, x1 = trX(float64(ix)), trX(float64(ix+1))+1
			shade := uint8(255 * data[j])
			colour := color.RGBA{shade, shade, shade, 255}
			pts[0].X, pts[0].Y = x0, y0
			pts[1].X, pts[1].Y = x0, y1
			pts[2].X, pts[2].Y = x1, y1
			pts[3].X, pts[3].Y = x1, y0
			c.FillPolygon(colour, pts)
			j++
		}
	}
}

// Image implements the DataRanger interface to set the scale
func (i *Image) DataRange() (xmin, xmax, ymin, ymax float64) {
	return 0, float64(i.Cols()), 0, float64(i.Rows())
}
