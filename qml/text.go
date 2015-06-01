package qml

import (
	"code.google.com/p/freetype-go/freetype"
	"code.google.com/p/freetype-go/freetype/truetype"
	"fmt"
	"github.com/gonum/plot/vg"
	"gopkg.in/qml.v1/gl/2.1"
	"gopkg.in/qml.v1/gl/glbase"
	"image"
	"io/ioutil"
	"os"
	"path/filepath"
)

const (
	lowChar  = 32
	highChar = 127
)

var (
	DefaultFontName  = "NimbusSanL-Bold.ttf"
	DefaultFontScale = 14
)

// A Glyph describes metrics for a single font glyph.
type glyph struct {
	x, adv int
}

// A Font allows rendering of text to an OpenGL context.
type Font struct {
	texture  glbase.Texture
	listbase uint32
	chars    []glyph
	height   int
}

// convert points to screen coords
func (ps *Plots) ptx(x int) float32 {
	return (2 / (xmax - xmin)) * (float32(x) / float32(ps.width))
}

func (ps *Plots) pty(y int) float32 {
	return -(2 / (ymax - ymin)) * (float32(y) / float32(ps.height))
}

// returns height of the current font in model coords
func (ps *Plots) TextHeight() float32 {
	return ps.pty(ps.font.height)
}

// returns the width of string in model coords
func (ps *Plots) TextWidth(s string) float32 {
	w := 0
	for _, ch := range s {
		w += ps.font.chars[ch-lowChar].adv
	}
	return ps.ptx(w)
}

func fontPath(name string) (string, error) {
	for _, d := range vg.FontDirs {
		p := filepath.Join(d, name)
		if _, err := os.Stat(p); err != nil {
			continue
		}
		return p, nil
	}
	return "", fmt.Errorf("Failed to locate a font file %s", name)
}

// LoadFont loads a truetype font from the given stream and applies the given font scale in points.
func (ps *Plots) loadFont(gl *GL.GL, fontName string, scale int) (*Font, error) {
	fontFile, err := fontPath(fontName)
	if err != nil {
		return nil, err
	}
	r, err := os.Open(fontFile)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	ttf, err := truetype.Parse(data)
	if err != nil {
		return nil, err
	}

	// get metrics
	f := new(Font)
	count := highChar - lowChar + 1
	glyphs := make([]glyph, count)
	gb := ttf.Bounds(int32(scale))
	gw := int(gb.XMax - gb.XMin)
	gh := int(gb.YMax - gb.YMin)
	f.height = gh

	// draw chars to image
	imgX, imgY := pow2(count*gw), pow2(gh)
	img := image.NewRGBA(image.Rect(0, 0, imgX, imgY))
	c := freetype.NewContext()
	c.SetDPI(72)
	c.SetFont(ttf)
	//c.SetHinting(freetype.FullHinting)
	c.SetFontSize(float64(scale))
	c.SetClip(img.Bounds())
	c.SetDst(img)
	c.SetSrc(image.White)
	gx := 0
	ch := rune(lowChar)
	for i := range glyphs {
		index := ttf.Index(ch)
		metric := ttf.HMetric(int32(scale), index)
		glyphs[i].adv = int(metric.AdvanceWidth)
		glyphs[i].x = int(gx)
		pt := freetype.Pt(int(gx), imgY-5)
		c.DrawString(string(ch), pt)
		gx += gw
		ch++
	}
	f.chars = glyphs

	// create texture
	tex := gl.GenTextures(1)
	gl.BindTexture(GL.TEXTURE_2D, tex[0])
	gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST)
	gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST)
	gl.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA, imgX, imgY, 0, GL.RGBA, GL.UNSIGNED_BYTE, img.Pix)
	f.texture = tex[0]
	f.listbase = gl.GenLists(int32(count))
	ghf := ps.pty(gh)
	thf := float32(gh) / float32(imgY)

	for i, g := range glyphs {
		gwf := ps.ptx(g.adv)
		twf := float32(g.adv) / float32(imgX)
		txf := float32(g.x) / float32(imgX)
		gl.NewList(f.listbase+uint32(i), GL.COMPILE)
		gl.Begin(GL.QUADS)
		gl.TexCoord2f(txf, 1)
		gl.Vertex2f(0, -ghf)
		gl.TexCoord2f(txf+twf, 1)
		gl.Vertex2f(gwf, -ghf)
		gl.TexCoord2f(txf+twf, 1-thf)
		gl.Vertex2f(gwf, 0)
		gl.TexCoord2f(txf, 1-thf)
		gl.Vertex2f(0, 0)
		gl.End()
		gl.Translatef(gwf, 0, 0)
		gl.EndList()
	}
	if errno := gl.GetError(); errno != GL.NO_ERROR {
		return nil, fmt.Errorf("GL error %d loading fonts\n", errno)

	}
	return f, nil
}

// DrawText method draws the given string at the specified coordinates.
func (f *Font) DrawText(gl *GL.GL, xpos, ypos, angle float32, str string) error {
	if len(str) == 0 {
		return nil
	}
	indices := []rune(str)
	for i := range indices {
		indices[i] -= lowChar
	}
	gl.PushAttrib(GL.ENABLE_BIT)
	gl.Enable(GL.TEXTURE_2D)
	gl.Enable(GL.BLEND)
	gl.BlendFunc(GL.SRC_ALPHA, GL.ONE_MINUS_SRC_ALPHA)
	gl.TexEnvf(GL.TEXTURE_ENV, GL.TEXTURE_ENV_MODE, GL.MODULATE)
	gl.BindTexture(GL.TEXTURE_2D, f.texture)

	gl.PushMatrix()
	gl.Translatef(xpos, ypos, 0)
	if angle != 0 {
		gl.Rotatef(angle, 0, 0, 1)
	}
	gl.ListBase(f.listbase)
	gl.CallLists(len(indices), GL.UNSIGNED_INT, indices)
	gl.PopMatrix()
	gl.PopAttrib()
	if errno := gl.GetError(); errno != GL.NO_ERROR {
		return fmt.Errorf("GL error %d drawing text\n", errno)
	}
	return nil
}

// pow2 returns the first power-of-two value >= to n.
// This can be used to create suitable texture dimensions.
func pow2(x int) int {
	x--
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	return int(x + 1)
}
