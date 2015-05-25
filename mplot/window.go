// Package mplot provides graphics plotting routines using GL and gonum/plot.
package mplot

import (
	"image"
	"image/color"

	"github.com/go-gl/gl/v2.1/gl"
	"github.com/go-gl/glfw/v3.1/glfw"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/gonum/plot/vg/vgimg"
)

type Window struct {
	*glfw.Window
	img    *image.RGBA
	canvas vg.CanvasSizer
	tex    uint32
	width  int32
	height int32
}

// NewWindow function opens a new on screen window of given size.
func NewWindow(width, height int, title string) (w *Window, err error) {

	// init opengl
	if err = glfw.Init(); err != nil {
		return
	}
	glfw.WindowHint(glfw.Resizable, glfw.False)
	glfw.WindowHint(glfw.ContextVersionMajor, 2)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	w = new(Window)
	if w.Window, err = glfw.CreateWindow(width, height, title, nil, nil); err != nil {
		return
	}
	w.MakeContextCurrent()
	if err = gl.Init(); err != nil {
		return
	}
	//version := gl.GoStr(gl.GetString(gl.VERSION))
	//fmt.Println("OpenGL version", version)

	// setup image and associated texture
	w.img = image.NewRGBA(image.Rect(0, 0, width, height))
	w.canvas = vgimg.NewImage(w.img)
	w.width, w.height = int32(width), int32(height)
	gl.Enable(gl.TEXTURE_2D)
	gl.GenTextures(1, &w.tex)
	gl.BindTexture(gl.TEXTURE_2D, w.tex)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w.width, w.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, nil)
	return
}

// Update method refreshes the screen
func (w *Window) Draw(rows, cols int, plt ...*Plot) {
	// draw plots onto image canvas
	width, height := w.canvas.Size()
	for i, p := range plt {
		// scale the canvas area
		row, col := i/cols, i%cols
		x0, y0 := float64(col)/float64(cols), float64(rows-row-1)/float64(rows)
		x1, y1 := float64(col+1)/float64(cols), float64(rows-row)/float64(rows)
		c := draw.Canvas{
			Canvas: w.canvas,
			Rectangle: draw.Rectangle{
				draw.Point{vg.Length(x0) * width, vg.Length(y0) * height},
				draw.Point{vg.Length(x1) * width, vg.Length(y1) * height},
			},
		}
		// rescale the axis limits
		for _, r := range p.ranges {
			p.X.Min, p.X.Max, p.Y.Min, p.Y.Max = r.DataRange()
		}
		p.Draw(c)
	}
	// refresh screen from image buffer
	gl.TexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w.width, w.height, gl.RGBA, gl.UNSIGNED_BYTE, gl.Ptr(w.img.Pix))
	gl.BindTexture(gl.TEXTURE_2D, w.tex)
	gl.Color4f(1, 1, 1, 1)
	gl.Begin(gl.QUADS)
	gl.Normal3f(0, 0, 1)
	gl.TexCoord2f(0, 1)
	gl.Vertex3f(-1, -1, 1)
	gl.TexCoord2f(1, 1)
	gl.Vertex3f(1, -1, 1)
	gl.TexCoord2f(1, 0)
	gl.Vertex3f(1, 1, 1)
	gl.TexCoord2f(0, 0)
	gl.Vertex3f(-1, 1, 1)
	gl.End()
	w.SwapBuffers()
	glfw.PollEvents()
}

type Plot struct {
	*plot.Plot
	ranges []plot.DataRanger
}

// New default plot with white on black background.
func New() *Plot {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	// set font
	titleFont, err := vg.MakeFont("Helvetica", 12)
	if err != nil {
		panic(err)
	}
	tickFont, err := vg.MakeFont("Helvetica", 10)
	if err != nil {
		panic(err)
	}
	p.Title.Font = titleFont
	p.X.Label.Font = titleFont
	p.Y.Label.Font = titleFont
	p.X.Tick.Label.Font = tickFont
	p.Y.Tick.Label.Font = tickFont
	p.Legend.Font = titleFont

	// set colors
	p.Title.Color = color.White
	p.X.Color = color.White
	p.Y.Color = color.White
	p.X.Label.Color = color.White
	p.Y.Label.Color = color.White
	p.X.Tick.Color = color.White
	p.Y.Tick.Color = color.White
	p.X.Tick.Label.Color = color.White
	p.Y.Tick.Label.Color = color.White
	p.Legend.Color = color.White
	p.BackgroundColor = color.Black

	// no padding between axis and grid
	p.X.Padding = 0
	p.Y.Padding = 0

	// legend style
	p.Legend.Top = true
	p.Legend.XOffs = vg.Points(-10)
	p.Legend.YOffs = vg.Points(-10)

	// add grid
	p.Add(plotter.NewGrid())
	return &Plot{Plot: p, ranges: []plot.DataRanger{}}
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
