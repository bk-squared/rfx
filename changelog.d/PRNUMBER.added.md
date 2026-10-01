Opt-in `forward(gradient="adjoint")` computes settled-spectrum design-permittivity
gradients on supported uniform Yee scenes using record-length-independent storage.
`ForwardResult.adjoint_settling` reports the final-to-peak electric-field amplitude
ratio on the design-box Yee edges; above about 1e-2 (-40 dB), the record has not settled.
Autodiff leaves this field `None`. Unsupported paths and design conductivity
overrides are refused; the conductivity derivative remains unvalidated (#1424 gCa).
