package scl

import (
	"fmt"
	"github.com/go-gl/cl/v1.2/cl"
	"reflect"
	"unsafe"
)

// Buffer type encapsulates a cl_mem structure to pass data between host and device
type Buffer struct {
	Buf  cl.Mem
	Ptr  unsafe.Pointer
	Size uint64
}

// NewBuffer allocates a new buffer with given mode and size in bytes.
// data may be nil in the case where we do not need to access the buffer from the host.
func NewBuffer(h *Hardware, mode cl.MemFlags, size int, data interface{}) *Buffer {
	var err cl.ErrorCode
	b := cl.CreateBuffer(h.Context, mode, cl.MemFlags(size), nil, &err)
	checkErr(err)
	return &Buffer{Buf: b, Ptr: getPointer(data), Size: uint64(size)}
}

// Write method writes host data to the device
func (b *Buffer) Write(h *Hardware) {
	err := cl.EnqueueWriteBuffer(h.Queue, b.Buf, cl.TRUE, 0, b.Size, b.Ptr, 0, nil, nil)
	checkErr(err)
}

// Read method reads memory back from device to host
func (b *Buffer) Read(h *Hardware) cl.ErrorCode {
	err := cl.EnqueueReadBuffer(h.Queue, b.Buf, cl.TRUE, 0, b.Size, b.Ptr, 0, nil, nil)
	if err == cl.SUCCESS {
		err = cl.Finish(h.Queue)
	}
	return err
}

// SetKernelArg method sets the buffer as an argument to a kernel
func (b *Buffer) setKernelArg(s *Software, argc uint32) cl.ErrorCode {
	return cl.SetKernelArg(s.Kernel, argc, 8, unsafe.Pointer(&b.Buf))
}

// Release method frees the buffer memory
func (b *Buffer) Release() {
	cl.ReleaseMemObject(b.Buf)
	b.Buf = nil
	b.Ptr = nil
	b.Size = 0
}

// String method gives a dump of the contents of the buffer
func (b *Buffer) String() string {
	s := fmt.Sprintf("<< size=%d: ", b.Size)
	if b.Ptr != nil {
		p := uintptr(b.Ptr)
		for i := 0; i < int(b.Size) && i < 32; i++ {
			s += fmt.Sprint(*(*byte)(unsafe.Pointer(p))) + " "
			p++
		}
		if b.Size >= 32 {
			s += "..."
		}
	}
	return s + ">>"
}

// convert from an interface to a pointer
func getPointer(ival interface{}) unsafe.Pointer {
	if ival == nil {
		return nil
	}
	v := reflect.ValueOf(ival)
	return unsafe.Pointer(v.Pointer())
}
