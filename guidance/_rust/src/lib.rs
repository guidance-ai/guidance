use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn engine_start(parser: &str, grammar: &str, ensure_bos_token: i32) -> PyResult<String> {
    Ok(format!("You passed {} and {}", parser, grammar))
}

#[pymodule]
fn guidancerust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_start, m)?)?;
    Ok(())
}
