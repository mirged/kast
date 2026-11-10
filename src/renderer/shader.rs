//! WGSL shader construction.
//!
//! Every shader in the crate is written as a fragment (or pass-specific) module that assumes the
//! helpers in `shaders/common.wgsl` are already in scope: BRDF terms, depth reconstruction, sky
//! colour, tonemap curves, hashing and noise. [`create`] prepends that common library and compiles
//! the result, so the shared code exists once on disk and once per pipeline in the binary — which
//! is what allows a single shader source file to be reused by several passes with different entry
//! points.

/// Compile `source` with `common.wgsl` prepended, labelled with `label` for validation messages.
///
/// The concatenation happens at compile time for the common part (`include_str!`), so the WGSL
/// text is part of the binary and nothing is read from disk at run time — required for a published
/// crate, where the `shaders/` directory may not exist beside the user's binary.
pub fn create(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    let combined = format!("{}\n{}", include_str!("../../shaders/common.wgsl"), source);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(combined.into()),
    })
}
