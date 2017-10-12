### mammoth

PyTorch implementation of [Maclaurin et al, 2015](https://arxiv.org/pdf/1502.03492.pdf)

Research code under active development.  Open issues if necessary.

#### Notes

Currently relies on a custom build of `pytorch` that uses deterministic operations -- out of the box, `cudnn` has some operations that are non-deterministic, which cause problems and need to be disabled.