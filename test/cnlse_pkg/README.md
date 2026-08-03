# Vendored `cnlse` package

Source: https://github.com/WUST-FOG/cgnlse-python (MIT license, see LICENSE).

Only the `cnlse/` subpackage is vendored here (not published on PyPI). Used by
`test/generate_vectorial_reference_data.py` as an independent, peer-reviewed
third-party reference for the Scenario 6 vectorial/birefringent adversarial
test in `test/test_adversarial.jl`. It is built directly on the real `gnlse`
package (same one used for Scenarios 1-5), following the same FFT/frequency
convention as Soliton.
