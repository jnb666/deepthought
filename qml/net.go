package qml

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"gopkg.in/qml.v1"
	"gopkg.in/qml.v1/gl/2.1"
	"image/color"
)

const (
	netPadX = 4
	netPadY = 4
)

// Network type is used to visualise the neural net.
type Network struct {
	qml.Object
	ctrl       *Ctrl
	Index      int
	Background color.RGBA
	Color      color.RGBA
	font       Font
	width      int
	height     int
}

// Step to previous entry in test set
func (n *Network) Prev() {
	n.Index--
	if n.Index < 0 {
		n.Index = n.ctrl.testData.NumSamples - 1
	}
	n.Call("update")
}

// Step to next entry in test set
func (n *Network) Next() {
	n.Index++
	if n.Index >= n.ctrl.testData.NumSamples {
		n.Index = 0
	}
	n.Call("update")
}

// convert points to screen coords
func (n *Network) ptx(x int) float32 {
	return 2 * float32(x) / float32(n.width)
}

func (n *Network) pty(y int) float32 {
	return -2 * float32(y) / float32(n.height)
}

// Paint method draws the network.
func (n *Network) Paint(paint *qml.Painter) {
	gl := GL.API(paint)
	n.width, n.height = n.Int("width"), n.Int("height")
	if n.font == 0 {
		n.font = loadFont(gl, DefaultFontName, DefaultFontScale, n.ptx, n.pty)
	}
	// x,y from -1 to +1
	setView(gl, n.width, n.height, 0, 0, 1, 1)
	r, g, b, a := col(n.Background)
	gl.ClearColor(r, g, b, a)
	gl.Clear(GL.COLOR_BUFFER_BIT)
	setColor(gl, n.Color)
	// run the network
	net := n.ctrl.network
	input := n.ctrl.testData.Input.Row(n.Index, n.Index+1)
	output := net.FeedForward(input)
	// update title with prediction
	out := net.Classify(output).Data(blas.ColMajor)
	exp := n.ctrl.testData.Classes.Row(n.Index, n.Index+1).Data(blas.ColMajor)
	title := fmt.Sprintf("Output = %g  Expected = %g", out[0], exp[0])
	n.font.DrawText(gl, -n.ptx(n.font.Size(title))/2, -0.98, 0, title)
	// get total size
	var xsize, ysize float32
	for _, layer := range net.Nodes {
		nx, ny := dimxy(layer.Dims())
		xsize += float32(nx + netPadX)
		ysize = fmax(ysize, float32(ny+netPadY))
	}
	maxSize := fmax(xsize, ysize)
	// draw the network
	var xpos float32
	for _, layer := range net.Nodes {
		nx, ny := dimxy(layer.Dims())
		dx := float32(nx+netPadX) / xsize
		gl.PushMatrix()
		gl.Scalef(1/maxSize, 1/maxSize, 1)
		gl.Translatef(maxSize*(2*xpos+dx-1), 0, 0)
		n.drawLayer(gl, nx, ny, layer.Values().Data(blas.ColMajor))
		gl.PopMatrix()
		xpos += dx
	}
	// update label
	n.ctrl.testLabel.Set("text", fmt.Sprintf("test: %d / %d", n.Index, n.ctrl.testData.NumSamples))
}

// draw one layer where each cell is unit size centered at the origin
func (n *Network) drawLayer(gl *GL.GL, nx, ny int, values []float64) {
	// draw activations
	gl.PushAttrib(GL.CURRENT_BIT)
	fx, fy := float32(nx), float32(ny)
	ypos := -fy
	for y := 0; y < ny; y++ {
		xpos := -fx
		for x := 0; x < nx; x++ {
			ix := y*nx + x
			if values[ix] >= 0 {
				gl.Color4f(0, float32(values[ix]), 0, 1)
			} else {
				gl.Color4f(-float32(values[ix]), 0, 0, 1)
			}
			gl.Begin(GL.QUADS)
			gl.Vertex2f(xpos, ypos)
			gl.Vertex2f(xpos+2, ypos)
			gl.Vertex2f(xpos+2, ypos+2)
			gl.Vertex2f(xpos, ypos+2)
			gl.End()
			xpos += 2
		}
		ypos += 2
	}
	gl.PopAttrib()
	// draw outline
	gl.Begin(GL.LINE_LOOP)
	px, py := n.ptx(1), n.pty(1)
	gl.Vertex2f(-fx-px, -fy-py)
	gl.Vertex2f(fx+px, -fy-py)
	gl.Vertex2f(fx+px, fy+py)
	gl.Vertex2f(-fx-px, fy+py)
	gl.End()
}

func dimxy(dims []int) (nx, ny int) {
	ny = dims[0]
	nx = 1
	if len(dims) > 1 {
		nx = dims[1]
	}
	return
}
