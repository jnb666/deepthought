package m32

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 1.1, 2.2, 3.3)
	t.Logf("\n%s\n", m)
}

func TestRandom(t *testing.T) {
	m := New(2, 3).Random(-0.5, 0.5)
	t.Logf("\n%s\n", m)
}

func TestTranspose(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 0, 1, 2, 3, 4, 5)
	m.Transpose()
	t.Logf("\n%s\n", m)
	expect := New(3, 2).Load(RowMajor, 0, 3, 1, 4, 2, 5)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestApply(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 1.1, 2.2, 3.3)
	m.Apply(m, func(x float32) float32 { return 2 * x })
	t.Logf("\n%s\n", m)
	expect := New(2, 3).Load(RowMajor, 2.2, 4.4, 6.6)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestMul(t *testing.T) {
	a := New(2, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6)
	t.Logf("a\n%s\n", a)
	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	t.Logf("b\n%s\n", b)
	m := New(2, 2).Mul(a, b, false)
	t.Logf("m\n%s\n", m)
	expect := New(2, 2).Load(RowMajor, 58, 64, 139, 154)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
	a.Transpose()
	m.Mul(a, b, true)
	t.Logf("m [trans]\n%s\n", m)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestMaxCol(t *testing.T) {
	m := New(5, 3).Load(RowMajor, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1)
	t.Logf("\n%s\n", m)
	c := New(5, 1).MaxCol(m)
	t.Logf("\n%s\n", c)
	expect := New(5, 1).Load(RowMajor, 0, 1, 2, 0, 2)
	if !reflect.DeepEqual(c, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}
