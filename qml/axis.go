package qml

import (
	"fmt"
	"github.com/jnb666/deepthought/vec"
	"gopkg.in/qml.v1/gl/2.1"
	"math"
)

const (
	maxTicks        = 6
	tickLength      = 0.015
	tickLabelOffset = 0.02
	axisLabelOffset = 0.05
)

// Axis type represents the axis for the plot
type Axis struct {
	Title     string
	FloatZero bool
	min       float32
	max       float32
	spacing   float32
	horiz     bool
}

func newAxis(horiz bool) *Axis {
	ax := new(Axis)
	ax.horiz = horiz
	ax.Rescale(0, 1)
	return ax
}

func (ax *Axis) paint(gl *GL.GL, ps *Plots) {
	p := ps.plt[ps.current]
	gl.LineWidth(1)
	// draw axis
	if ax.horiz {
		drawLine(gl, 0, 0, 1, 0)
	} else {
		drawLine(gl, 0, 0, 0, 1)
	}
	// draw axis ticks
	val := ax.min
	yoff := ps.TextHeight() / 2
	for val <= ax.max {
		label := fmt.Sprintf("%.4g", val)
		width := ps.TextWidth(label)
		xpos, ypos := p.Trans(val, val)
		if ax.horiz {
			drawLine(gl, xpos, 0, xpos, -tickLength)
			ps.font.DrawText(gl, xpos-width/2, -tickLabelOffset, 0, label)
		} else {
			drawLine(gl, 0, ypos, -tickLength, ypos)
			ps.font.DrawText(gl, -width-tickLabelOffset, ypos+yoff, 0, label)
		}
		val += ax.spacing
	}
	// draw grid
	if ps.Grid {
		gl.PushAttrib(GL.CURRENT_BIT)
		setColor(gl, ps.GridColor)
		val := ax.min + ax.spacing
		for val <= ax.max {
			xpos, ypos := p.Trans(val, val)
			if ax.horiz {
				drawLine(gl, xpos, 0, xpos, 1)
			} else {
				drawLine(gl, 0, ypos, 1, ypos)
			}
			val += ax.spacing
		}
		gl.PopAttrib()
	}
	// axis labels
	if ax.Title != "" {
		if ax.horiz {
			ps.font.DrawText(gl, 1-ps.TextWidth(ax.Title), -axisLabelOffset, 0, ax.Title)
		} else {
			ps.font.DrawText(gl, -axisLabelOffset-ps.TextHeight(), 1-ps.TextWidth(ax.Title), 90, ax.Title)
		}
	}
}

// Rescale method resets the axis scale
func (ax *Axis) Rescale(min, max float32) {
	if max-min < epsilon {
		min, max = 0, 1
	}
	if !ax.FloatZero && min > 0 {
		min = 0
	}
	axisRange := vec.Nicenum(float64(max-min), false)
	spacing := vec.Nicenum(axisRange/float64(maxTicks-1), true)
	ax.spacing = float32(spacing)
	ax.min = float32(math.Floor(float64(min)/spacing) * spacing)
	ax.max = float32(math.Ceil(float64(max)/spacing) * spacing)
}
