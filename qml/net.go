package qml

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"gopkg.in/qml.v1"
	"gopkg.in/qml.v1/gl/2.1"
	"image/color"
)

const (
	netPadX   = 6
	netPadY   = 6
	grid      = 4
	grid2     = grid * grid
	gridScale = 0.98
)

// Network type is used to visualise the neural net.
type Network struct {
	qml.Object
	ctrl       *Ctrl
	label      qml.Object
	Background color.RGBA
	Color      color.RGBA
	font       *Font
	width      int
	height     int
	errorsOnly bool
	filter     int
	perPage    int
	next       int
	res        []results
}

type results struct {
	index  int
	output int
	target int
	input  []float64
}

// convert points to screen coords
func (n *Network) ptx(x int) float32 {
	return 2 * float32(x) / float32(n.width)
}

func (n *Network) pty(y int) float32 {
	return -2 * float32(y) / float32(n.height)
}

// Step to first entry in test set
func (n *Network) First() {
	n.next = 0
	n.res = []results{}
	n.Run(1)
}

// Step to previous entry in test set
func (n *Network) Prev() {
	if n.res != nil && len(n.res) > 0 {
		n.next = n.res[0].index
	}
	n.next--
	n.Run(-1)
}

// Step to next entry in test set
func (n *Network) Next() {
	if n.res != nil && len(n.res) > 0 {
		n.next = n.res[len(n.res)-1].index
	}
	n.next++
	n.Run(1)
}

// Filter to only show errors
func (n *Network) Filter(on bool, val int) {
	n.errorsOnly = on
	n.filter = val
	n.next = 0
	n.First()
}

// Enable compact view
func (n *Network) Compact(on bool) {
	if on {
		n.perPage = grid2
	} else {
		n.perPage = 1
	}
	n.next = 0
	n.First()
}

// Run network. Step to next data entry if filter is not matched.
func (n *Network) Run(offset int) {
	net := n.ctrl.network
	data := n.ctrl.testData
	if n.perPage == 0 {
		n.perPage = 1
	}
	got := 0
	done := false
	for try := 0; try < data.NumSamples; try++ {
		n.next = (n.next + data.NumSamples) % data.NumSamples
		exp := data.Classes.Row(n.next, n.next+1).Data(blas.ColMajor)
		if n.filter < 0 || int(exp[0]) == n.filter {
			input := data.Input.Row(n.next, n.next+1)
			output := net.FeedForward(input)
			out := net.Classify(output).Data(blas.ColMajor)
			if !n.errorsOnly || out[0] != exp[0] {
				res := results{
					index:  n.next,
					input:  input.Data(blas.ColMajor),
					output: int(out[0]),
					target: int(exp[0]),
				}
				if got, done = n.addResult(got, res, offset < 0); done {
					break
				}
			}
		}
		n.next += offset
		if (n.next < 0 || n.next >= data.NumSamples) && got > 0 {
			break
		}
	}
	n.Call("update")
}

func (n *Network) addResult(got int, res results, prepend bool) (int, bool) {
	for _, r := range n.res {
		if r.index == res.index {
			return got, true
		}
	}
	if prepend {
		end := len(n.res)
		if len(n.res) >= n.perPage {
			end--
		}
		n.res = append([]results{res}, n.res[:end]...)
	} else {
		start := 0
		if len(n.res) >= n.perPage {
			if got == 0 {
				start = len(n.res)
			} else {
				start = 1
			}
		}
		n.res = append(n.res[start:], res)
	}
	got++
	return got, got >= n.perPage
}

// Paint method draws the network.
func (n *Network) Paint(paint *qml.Painter) {
	gl := GL.API(paint)
	n.width, n.height = n.Int("width"), n.Int("height")
	if n.font == nil {
		n.font = loadFont(gl, DefaultFontName, DefaultFontScale, n.ptx, n.pty)
	}
	// x,y from -1 to +1
	setView(gl, n.width, n.height, 0, 0, 1, 1)
	r, g, b, a := col(n.Background)
	gl.ClearColor(r, g, b, a)
	gl.Clear(GL.COLOR_BUFFER_BIT)
	setColor(gl, n.Color)
	if n.res == nil || len(n.res) == 0 {
		return
	}
	n.ctrl.testLabel.Set("text", fmt.Sprintf("test: %d / %d", n.res[0].index, n.ctrl.testData.NumSamples))

	if n.perPage == 1 {
		// detail view with one test sample per page
		var xsize, ysize float32
		for _, layer := range n.ctrl.network.Nodes {
			nx, ny := dimxy(layer.Dims())
			xsize += float32(nx + netPadX)
			ysize = fmax(ysize, float32(ny+netPadY))
		}
		maxSize := fmax(xsize, ysize)
		var xpos float32
		for _, layer := range n.ctrl.network.Nodes {
			nx, ny := dimxy(layer.Dims())
			dx := float32(nx+netPadX) / xsize
			gl.PushMatrix()
			gl.Scalef(1/maxSize, 1/maxSize, 1)
			gl.Translatef(maxSize*(2*xpos+dx-1), 0, 0)
			n.drawLayer(gl, nx, ny, layer.Values().Data(blas.ColMajor))
			gl.PopMatrix()
			xpos += dx
		}
		title := fmt.Sprintf("Expected = %d   Output = %d", n.res[0].target, n.res[0].output)
		n.font.DrawText(gl, -n.ptx(n.font.Size(title))/2, -0.98, 0, title)
	} else {
		// compact grid view with just input and output
		nx, ny := dimxy(n.ctrl.network.Nodes[0].Dims())
		maxSize := fmax(float32(nx+netPadX), float32(ny+netPadY))
		for iy := 0; iy < grid; iy++ {
			for ix := 0; ix < grid; ix++ {
				i := iy*grid + ix
				if i >= len(n.res) {
					return
				}
				xpos, ypos := float32(2*ix)-grid+1, float32(2*iy)-grid+1
				gl.PushMatrix()
				gl.Scalef(1.0/grid, 1.0/grid, 1)
				gl.Translatef(xpos, ypos, 0)
				gl.Scalef(gridScale/maxSize, gridScale/maxSize, 1)
				n.drawLayer(gl, nx, ny, n.res[i].input)
				gl.PopMatrix()
				title := fmt.Sprintf("%d: %d => %d", n.res[i].index, n.res[i].target, n.res[i].output)
				width := n.ptx(n.font.Size(title))
				n.font.DrawText(gl, (xpos+1-gridScale)/grid-width/2, (ypos-gridScale)/grid, 0, title)
			}
		}
	}
}

// draw one layer where each cell is unit size centered at the origin
func (n *Network) drawLayer(gl *GL.GL, nx, ny int, values []float64) {
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
			drawBox(gl, xpos, ypos, xpos+2, ypos+2, true)
			xpos += 2
		}
		ypos += 2
	}
	gl.PopAttrib()
	px, py := n.ptx(1), n.pty(1)
	drawBox(gl, -fx-px, -fy-py, fx+px, fy+py, false)
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
